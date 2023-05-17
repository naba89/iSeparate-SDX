"""Copyright: Nabarun Goswami (2023)."""
import os
from functools import partial
from multiprocessing import Pool

from pydub import AudioSegment
from pydub.silence import split_on_silence

from tqdm import tqdm


def remove_silence(song_name, indir, instrs, outdir):
    for instr in instrs:
        audio_file = os.path.join(indir, song_name, f'{instr}.wav')
        sound = AudioSegment.from_wav(audio_file)
        if instr == 'sfx':
            output_dir = os.path.join(outdir, song_name)
            os.makedirs(output_dir, exist_ok=True)
            outfile = os.path.join(output_dir, f'{instr}.wav')
            sound.export(outfile)
            continue

        audio_chunks = split_on_silence(sound
                                        , min_silence_len=1000
                                        , silence_thresh=-80
                                        , keep_silence=False
                                        )

        # Putting the file back together
        try:
            combined = AudioSegment.empty()
            for i, chunk in enumerate(audio_chunks):
                if instr == 'music':
                    if i == 0:
                        combined = chunk
                    else:
                        if len(chunk) < 100 or len(combined) < 100:
                            combined += chunk
                        else:
                            combined = combined.append(chunk, crossfade=100)
                else:
                    combined += chunk
        except ValueError as e:
            print(f'Short chunk: {audio_file}')
            print(e)
            raise e
        output_dir = os.path.join(outdir, song_name)
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, f'{instr}.wav')
        combined.export(outfile)


if __name__ == '__main__':
    indir1 = 'DATASETS/DnR/dnr_v2/tr'
    outdir = 'DATASETS/DnR/tr_silence_removed'
    instruments = ['music', 'sfx', 'speech']

    songs1 = os.listdir(indir1)

    # filter non directories
    songs1 = [s for s in songs1 if os.path.isdir(os.path.join(indir1, s))]

    # filter .ipynb_checkpoints
    songs1 = [s for s in songs1 if not s.startswith('.')]

    nproc = min(os.cpu_count(), 32)
    print(f'Processing {len(songs1)} songs in {indir1} with {nproc} processes')
    with Pool(nproc) as p:
        _ = list(tqdm(p.imap(partial(remove_silence,
                                     indir=indir1,
                                     outdir=outdir,
                                     instrs=instruments), songs1), total=len(songs1)))
