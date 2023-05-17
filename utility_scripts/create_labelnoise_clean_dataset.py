"""Copyright: Nabarun Goswami (2023)."""
import csv
import os
import shutil

from tqdm import tqdm

if __name__ == '__main__':

    basedir = 'DATASETS/SDX2023/moisesdb23_labelnoise_v1.0'
    outdir = 'DATASETS/SDX2023/dwt_labelnoise_clean_v2'

    with open('file_lists/moisesdb23_labelnoise_v1.0_sep_dwt_scores.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    data = data[1:]

    threshold_min = 10
    threshold_max = 1000

    for i in tqdm(range(len(data))):
        song, bass, drums, other, vocals = data[i]
        songdir = f'{outdir}/{song}'
        if threshold_min <= float(bass) < threshold_max:
            outdir_bass = f'{songdir}/bass.wav'
            os.makedirs(songdir, exist_ok=True)
            shutil.copy(f'{basedir}/{song}/bass.wav', outdir_bass)
        if threshold_min <= float(drums) < threshold_max:
            outdir_drums = f'{songdir}/drums.wav'
            os.makedirs(songdir, exist_ok=True)
            shutil.copy(f'{basedir}/{song}/drums.wav', outdir_drums)
        if threshold_min <= float(other) < threshold_max:
            outdir_other = f'{songdir}/other.wav'
            os.makedirs(songdir, exist_ok=True)
            shutil.copy(f'{basedir}/{song}/other.wav', outdir_other)
        if threshold_min <= float(vocals) < threshold_max:
            outdir_vocals = f'{songdir}/vocals.wav'
            os.makedirs(songdir, exist_ok=True)
            shutil.copy(f'{basedir}/{song}/vocals.wav', outdir_vocals)
