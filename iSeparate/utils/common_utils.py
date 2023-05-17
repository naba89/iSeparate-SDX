import argparse
import glob
import os

import torch
import yaml


class ParseFromConfigFile(argparse.Action):
    def __init__(self, option_strings, type, dest, help=None, required=False):
        super(ParseFromConfigFile, self).__init__(
            option_strings=option_strings, type=type, dest=dest, help=help, required=required
        )

    def __call__(self, parser, namespace, values, option_string):
        with open(values, "r") as f:
            data = yaml.safe_load(f)

        for k, v in data.items():
            setattr(namespace, k, v)


def delete_older_checkpoints(directory, keep=5):
    files = list(glob.glob(directory + "/*.pt"))
    files = [f for f in files if "last" not in f and "best" not in f]
    sorted_checkpoints = sorted(files, key=os.path.getctime, reverse=True)[keep:]
    for f in sorted_checkpoints:
        if "best" in f:
            continue
        os.remove(f)


def save_best_state(best_state, args, val_loss, output_dir, model_name, iteration):
    checkpoint = {
        "state_dict": best_state,
        "config": args,
        "val_loss": val_loss,
        "iteration": iteration,
    }
    checkpoint_filename = "checkpoint_{}_best.pt".format(model_name)
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    print("Saving best models state to {}".format(checkpoint_path))

    torch.save(checkpoint, checkpoint_path)
    return


def save_checkpoint(
        iteration,
        model,
        optimizer,
        epoch,
        config,
        output_dir,
        model_name,
        best_val_loss=None,
        emas=None,
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = {
        "iteration": iteration,
        "epoch": epoch,
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    if emas is not None:
        for kind, _emas in emas.items():
            for k, ema in enumerate(_emas):
                checkpoint[f'ema_{kind}_{k}'] = ema.state_dict()

    checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, iteration)
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)

    torch.save(checkpoint, checkpoint_path)

    symlink_src = checkpoint_path
    symlink_dst = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.islink(os.path.abspath(symlink_dst)):
        os.remove(os.path.abspath(symlink_dst))

    os.symlink(os.path.abspath(symlink_src), os.path.abspath(symlink_dst))
    delete_older_checkpoints(output_dir)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(filepath, model, optimizer, epoch, emas=None, map_location='cpu'):
    checkpoint = torch.load(filepath, map_location=map_location)
    epoch[0] = checkpoint["epoch"] + 1
    config = checkpoint["config"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    iteration = checkpoint["iteration"] + 1
    if emas is not None:
        for kind, _emas in emas.items():
            for k, ema in enumerate(_emas):
                if f'ema_{kind}_{k}' in checkpoint:
                    ema.load_state_dict(checkpoint[f'ema_{kind}_{k}'])
                else:
                    ema.lazy_init(model)

    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint["best_val_loss"]
    else:
        best_val_loss = None

    return model, optimizer, epoch, iteration, config, best_val_loss, emas
