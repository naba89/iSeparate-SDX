"""Copyright: Nabarun Goswami (2023)."""
import os
import shutil


def create_subset(subset_name, data_root_dir, output_dir):
    with open(f'file_lists/{subset_name}.txt', 'r') as f:
        data = f.readlines()

    data = [x.strip() for x in data]

    os.makedirs(f'{output_dir}/{subset_name}', exist_ok=True)
    for line in data:
        inp_file = f'{data_root_dir}/{line}'
        out_file = f'{output_dir}/{subset_name}/{line}'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copytree(inp_file, out_file)


if __name__ == '__main__':

    root_dir = 'DATASETS/DnR/dnr_v2/cv'
    outdir = 'DATASETS/DnR/dnr_v2'

    subset1 = 'sdx_valid'

    create_subset(subset1, root_dir, outdir)
