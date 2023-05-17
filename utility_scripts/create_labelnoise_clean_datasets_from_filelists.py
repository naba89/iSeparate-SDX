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
        shutil.copy(inp_file, out_file)


if __name__ == '__main__':

    labelnoise_root_dir = 'DATASETS/SDX2023/moisesdb23_labelnoise_v1.0'
    outdir = 'DATASETS/SDX2023'

    subset1 = 'labelnoise_clean_v2'
    subset2 = 'dwt_labelnoise_clean_v2'

    create_subset(subset1, labelnoise_root_dir, outdir)
    create_subset(subset2, labelnoise_root_dir, outdir)

