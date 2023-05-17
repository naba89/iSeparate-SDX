"""Copyright: Nabarun Goswami (2023)."""
import argparse
import importlib
import numpy as np
from collections import OrderedDict

import torch
import torch.distributed as dist
import tqdm
import os
import torchaudio

from iSeparate.datasets import datapipes
from iSeparate.losses.loss_switcher import get_eval_metric
from iSeparate.utils.common_utils import ParseFromConfigFile


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


def evaluate(args, models, eval_loader, device_id, disable_pbar, rank, world_size):
    target_sources = models[0].target_sources
    target_indices = models[0].target_indices

    eval_metrics_all = [None] * world_size
    eval_metrics_local = []
    song_names = [None] * world_size
    song_names_local = []

    loss_fn = get_eval_metric(args.eval_metric)
    sr = args.eval_params["sr"]

    count = torch.zeros(1, device=device_id)
    with torch.inference_mode():
        for batch in tqdm.tqdm(eval_loader, dynamic_ncols=True,
                               disable=disable_pbar):
            sources, song_name, *_ = batch
            sources = sources.to(device_id)
            mixture_wav = sources.sum(1)
            target_wav = sources[:, target_indices]

            mean = mixture_wav.mean(1).mean()
            std = mixture_wav.mean(1).std()
            mixture_wav = (mixture_wav - mean) / std
            output_wav = 0
            for model in models:
                output_wav += model.separate(mixture_wav, disable_pbar=disable_pbar,
                                             **args.eval_params).detach().cpu()
            output_wav /= len(models)
            output_wav = output_wav * std.cpu() + mean.cpu()

            loss = loss_fn(output_wav, target_wav.cpu())
            if rank == 0:
                print(loss)
            eval_metrics_local.append(loss.detach().cpu())
            song_names_local.append(song_name[0])

            if args.save_eval_wavs:
                out_dir = os.path.join(args.output, "eval_wavs", song_name[0])
                os.makedirs(out_dir, exist_ok=True)
                for i, instr in enumerate(target_sources):
                    torchaudio.save(f"{out_dir}/{instr}.wav",
                                    output_wav[0, i].detach().cpu(), sr, encoding="PCM_S")

            count += 1
    # dist.barrier()
    dist.all_gather_object(eval_metrics_all, eval_metrics_local)
    dist.all_gather_object(song_names, song_names_local)
    eval_metrics_all = [item for sublist in eval_metrics_all for item in sublist]
    song_names = [item for sublist in song_names for item in sublist]

    # create a csv file with song names and eval metrics for each instrument
    if rank == 0:
        with open(os.path.join(args.output, "eval_metrics.csv"), "w") as f:
            f.write("song_name")
            for instr in target_sources:
                f.write(f",{instr}")
            f.write("\n")
            for song, metric in zip(song_names, eval_metrics_all):
                f.write(song)
                for m in metric:
                    f.write(f",{m}")
                f.write("\n")

    eval_metrics_all = torch.stack(eval_metrics_all, dim=0)
    if args.reduction == "mean":
        eval_metrics_all = eval_metrics_all.mean(0)
    elif args.reduction == "median":
        eval_metrics_all = eval_metrics_all.median(0)

    val_losses = {}
    for metric, instr in zip(eval_metrics_all, target_sources):
        val_losses[instr] = -metric.item()
        if rank == 0:
            print(f"{instr}: {val_losses[instr]:.4f}")

    val_loss = sum(val_losses.values()) / len(val_losses)
    if rank == 0:
        print(f"Song: {val_loss:.4f}")
    return val_loss, val_losses


def main():
    parser = argparse.ArgumentParser(description="PyTorch Evaluation")
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    args.output = os.path.join(args.save_dir, args.model_name)

    dist.init_process_group("gloo")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}

    models = []
    for model_path in args.models:
        ckpt = torch.load(model_path, map_location=map_location)
        cfg = ckpt["config"]
        module = importlib.import_module(args.model_module)
        model = getattr(module, args.model_name)(**cfg.model_args).to(device_id)
        new_state_dict = OrderedDict()
        for k, v in ckpt[args.state_type].items():
            name = k.replace("module.", "")  # removing 'moldule.' from key
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model = model.eval()
        models.append(model)
        if local_rank == 0:
            print("Loaded model from iteration: {}".format(ckpt["iteration"]))

    test_loader_fn = args.data_loader_args.get("test_loader", "stem_folder_dataloader")
    testset = getattr(datapipes, test_loader_fn)(**args.data_loader_args["test"], batch_size=1, return_dp=True)

    testsets = testset.round_robin_demux(world_size)
    testset = testsets[local_rank]

    disable_pbar = local_rank != 0
    evaluate(args, models, testset, device_id, disable_pbar, local_rank, world_size)


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh evaluate_dist.py <path/to/config.yaml>
