import argparse
import bisect
import json
import os
import random

import torch
import torch.distributed as dist
import torchaudio
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

import iSeparate.datasets.datapipes as datapipes
from iSeparate.losses.loss_switcher import get_loss_fn, get_eval_metric
from iSeparate.models.model_switcher import get_model
from iSeparate.utils.common_utils import (
    ParseFromConfigFile,
    get_last_checkpoint_filename,
    load_checkpoint,
    save_checkpoint, save_best_state,
)
from iSeparate.utils.ema import ModelEMA2
from iSeparate.utils.tb_logger import TBLogger as Logger

torch.backends.cudnn.benchmark = True
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision('high')


def parse_args(parser):
    """
    Parse from config file.
    Add other arguments if needed
    """
    parser.add_argument(
        "--config-file", action=ParseFromConfigFile, type=str, help="Path to configuration file"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu-run", action="store_true")
    return parser


def evaluate(args, model, eval_loader, kind, device_id, disable_pbar, rank, world_size):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    mixing_sources = sorted(args.data_loader_args["train"]["mixing_sources"])
    target_sources = sorted(args.model_args["target_sources"])
    target_indices = [mixing_sources.index(s) for s in target_sources]

    val_losses = {instr: [] for instr in target_sources}
    loss_fn = get_eval_metric(args.eval_metric)
    sr = args.data_loader_args["train"]["sample_rate"]

    count = torch.zeros(1, device=device_id)
    with torch.inference_mode():
        for batch in tqdm.tqdm(eval_loader, desc=kind, dynamic_ncols=True,
                               disable=disable_pbar):
            sources, song_name, *_ = batch
            sources = sources.to(device_id)
            mixture_wav = sources.sum(1)
            target_wav = sources[:, target_indices]
            output_wav = model.separate(mixture_wav,
                                        patch_length=args.val_patch_length,
                                        hop_length=args.val_hop_length,
                                        sr=sr,
                                        use_window=args.use_window,
                                        disable_pbar=disable_pbar,
                                        shifts=getattr(args, 'val_shifts', 0),
                                        )

            loss = loss_fn(output_wav, target_wav)
            for _loss, instr in zip(loss, target_sources):
                val_losses[instr].append(_loss.detach())

            if args.save_eval_wavs:
                out_dir = os.path.join(args.output, "eval_wavs", kind, song_name[0])
                os.makedirs(out_dir, exist_ok=True)
                for i, instr in enumerate(target_sources):
                    torchaudio.save(f"{out_dir}/{instr}.wav",
                                    output_wav[0, i].detach().cpu(), sr)

            count += 1

    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    print_string = f""
    for k, v in val_losses.items():
        val_losses[k] = sum(val_losses[k]) / len(val_losses[k])
        dist.all_reduce(val_losses[k], op=dist.ReduceOp.SUM)
        val_losses[k] /= world_size
        print_string += f"{k}: {-val_losses[k]:.4f}, "

    val_loss = sum(val_losses.values()) / len(val_losses)
    print_string += f"Song: {-val_loss:.4f}"
    if rank == 0:
        print(print_string)
    return val_loss, val_losses


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    args.output = os.path.join(args.exp_dir, args.model_name, args.exp_name)

    args.device = "cuda" if torch.cuda.is_available() and not args.cpu_run else "cpu"

    dist.init_process_group("nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print(f"Start training {args.model_name}:{args.exp_name} on rank {local_rank}/{world_size}.")

    if local_rank == 0:
        print(json.dumps(vars(args), indent=4))

    if local_rank == 0:
        os.makedirs(args.output, exist_ok=True)
        tb_log_dir = os.path.join(args.output, "TB_Logs")
        os.makedirs(tb_log_dir, exist_ok=True)
        TBLogger = Logger(tb_log_dir)
    else:
        TBLogger = None

    model = get_model(
        model_name=args.model_name,
        model_args=args.model_args,
    ).to(device_id)

    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters: {}M".format(num_params / 1024 / 1024))
    emas = {'batch': [], 'epoch': []}
    for kind in args.ema.keys():
        decays = args.ema[kind]
        device = device_id if kind == 'batch' else 'cpu'
        if decays:
            for decay in decays:
                emas[kind].append(ModelEMA2(model, decay, device=device))

    opt_name = getattr(args, 'optimizer', 'Adam')

    optimizer = getattr(torch.optim, opt_name)(model.parameters(), lr=args.lr,
                                               **args.optimizer_args if hasattr(args, 'optimizer_args') else {})

    start_epoch = [0]
    iteration = 0
    best_val_loss = None

    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(args.output, args.model_name)

    if args.checkpoint_path != "":
        map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}
        model, optimizer, epoch, iteration, config, best_val_loss, emas = load_checkpoint(
            args.checkpoint_path, model, optimizer, start_epoch, emas, map_location
        )

        if not args.use_previous_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

    model = DDP(model, device_ids=[device_id])
    start_epoch = start_epoch[0]

    scheduler = None
    if args.scheduler is not None:
        last_epoch = start_epoch - 1
        scheduler = getattr(torch.optim.lr_scheduler, args.scheduler)(optimizer,
                                                                      **args.scheduler_params, last_epoch=last_epoch)

    criterion = get_loss_fn(args.loss_fn)

    assert args.batch_size % world_size == 0, "Batch size must be divisible by world size"
    batch_size = args.batch_size // world_size

    train_loader_fn = args.data_loader_args.get("train_loader", "stem_folder_dataloader")

    trainset = getattr(datapipes, train_loader_fn)(**args.data_loader_args["train"], batch_size=batch_size)
    train_loader = iter(trainset)

    valset = None
    val_loader = None
    skip_eval = getattr(args, "skip_eval", False)
    if not skip_eval:
        val_loader_fn = args.data_loader_args.get("val_loader", "stem_folder_dataloader")
        valset = getattr(datapipes, val_loader_fn)(**args.data_loader_args["validation"], batch_size=1)
        val_loader = valset

    model.train()

    best_model = "ema_batch_0"

    if local_rank != 0 or args.disable_pbar:
        disable_pbar = True
    else:
        disable_pbar = False

    with tqdm.trange(start_epoch, args.epochs, initial=start_epoch,
                     total=args.epochs, disable=disable_pbar,
                     dynamic_ncols=True) as t:
        for epoch in t:
            description = f'Epoch {epoch + 1} ({best_model}:{-best_val_loss if best_val_loss is not None else 0:.4f})'
            t.set_description(description)
            trainset.seed(epoch)
            with tqdm.trange(args.num_batches_per_epoch, leave=False, disable=disable_pbar,
                             dynamic_ncols=True) as t2:
                train_losses = []
                for _ in t2:
                    batch = next(train_loader)
                    model.train()
                    sources, *aux = batch
                    sources = sources.to(args.device, non_blocking=True)
                    outputs = model(sources)
                    if len(outputs) == 2:
                        output, target = outputs
                        sources = None
                    else:
                        output, target, sources = outputs

                    if 'mean_teacher' in args.loss_fn:
                        loss_weight_index = bisect.bisect_right(args.loss_weights_milestones, epoch)
                        mean_teacher_interval = getattr(args, 'mean_teacher_interval', 1)
                        mean_teacher_enable = (iteration % mean_teacher_interval) == 0
                        loss = criterion(output, target, emas['batch'][0].ema_model,
                                         args.loss_weights[loss_weight_index], mean_teacher_enable)
                    else:
                        loss = criterion(output, target)

                    flooding = getattr(args, 'flooding', 0)
                    if flooding > 0:
                        loss = (loss - flooding).abs() + flooding

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if 'grad_clip_norm' in args:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                    optimizer.step()

                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR) or \
                                isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                            scheduler.step()

                    for ema in emas['batch']:
                        ema.update(model)

                    train_losses.append(loss.detach())
                    iteration += 1

            train_losses = torch.stack(train_losses).mean()
            dist.all_reduce(train_losses, op=dist.ReduceOp.SUM)
            train_losses /= world_size
            if local_rank == 0:
                loss_dict = {"train_loss": train_losses.item()}
                lr = optimizer.param_groups[0]["lr"]
                TBLogger.log_training(loss_dict, lr, iteration)

            if scheduler is not None:
                if optimizer.param_groups[0]["lr"] > float(args.min_lr) and \
                        not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR) and \
                        not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            for ema in emas['epoch']:
                ema.update(model)

            if epoch % args.audio_log_interval == 0 and local_rank == 0:
                random_idx = random.randint(0, len(sources) - 1)

                y_wav = target['y'][random_idx, :, :1].detach()
                y_wav = y_wav / max(1.01 * y_wav.abs().max(), 1)

                y_hat_wav = output['y_hat'][random_idx, :, :1].detach()
                y_hat_wav = y_hat_wav / max(1.01 * y_hat_wav.abs().max(), 1)

                if sources is not None:
                    sources = sources[random_idx, :, :1].detach()
                    x_wav = sources.sum(0)
                else:
                    x_wav = y_wav.sum(0)

                x_wav = x_wav / max(1.01 * x_wav.abs().max(), 1)

                audios = {
                    "mixture": x_wav.cpu().numpy(),
                    "target": y_wav.cpu().numpy(),
                    "prediction": y_hat_wav.cpu().numpy(),
                    "target_names": sorted(args.model_args["target_sources"])
                }
                TBLogger.log_audios(audios, None)

            if (epoch+1) % args.eval_interval == 0 and not skip_eval:
                if local_rank == 0:
                    print(f"Evaluating model at epoch: {epoch+1}...")
                b_val_loss, b_val_losses = evaluate(args, model, val_loader, kind="current",
                                                    device_id=device_id, disable_pbar=disable_pbar,
                                                    rank=local_rank, world_size=world_size)
                best_state = model.state_dict()
                best_model = "current"
                for kind, _emas in emas.items():
                    for k, ema in enumerate(_emas):
                        val_loss, val_losses = evaluate(args, ema.ema_model, val_loader, kind=f'ema_{kind}_{k}',
                                                        device_id=device_id, disable_pbar=disable_pbar,
                                                        rank=local_rank, world_size=world_size)
                        if val_loss < b_val_loss:
                            b_val_loss = val_loss
                            b_val_losses = val_losses
                            best_state = ema.ema_model.state_dict()
                            best_model = f'ema_{kind}_{k}'
                val_loss_dict = {"avg_val_sdr": -b_val_loss}
                for k, v in b_val_losses.items():
                    val_loss_dict[f'{k}_val_sdr'] = -v
                if best_val_loss is None or b_val_loss < best_val_loss:
                    if local_rank == 0:
                        save_best_state(best_state, args, b_val_loss, args.output, args.model_name, iteration)
                    best_val_loss = b_val_loss

                if local_rank == 0:
                    TBLogger.log_training(val_loss_dict, None, iteration)

            if (epoch+1) % args.save_interval == 0 and local_rank == 0:
                save_checkpoint(
                    iteration,
                    model,
                    optimizer,
                    epoch + 1,
                    args,
                    args.output,
                    args.model_name,
                    best_val_loss,
                    emas=emas
                )
            if TBLogger is not None:
                TBLogger.flush()

    trainset.shutdown()
    if not skip_eval:
        valset.shutdown()


if __name__ == "__main__":
    main()

    # CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py <path/to/config.yaml>
