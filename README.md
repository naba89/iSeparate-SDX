# iSeparate-SDX

This library contains the code to train and reproduce the results for
the submissions by username: `subatomicseer`

**Author**: Nabarun Goswami

**Affiliation**: Harada-Osa-Mukuta-Kurose Lab, The University of Tokyo

## Submission and Results Summary

* MDX Leaderboard A (labelnoise)
    * Submission repo: [MDX2023-labelnoise-submission](https://gitlab.aicrowd.com/subatomicseer/MDX2023-labelnoise-submission)
	* Submission ID: 220423
	* Submitter: subatomicseer
	* Final rank: 2nd place
    * Final scores:
              
      |  SDR_song  |  SDR_bass  | SDR_drums  |  SDR_other  |  SDR_vocals  |
      |:----------:|:----------:|:----------:|:-----------:|:------------:|
      |   6.601    |   6.696    |   7.026    |    4.611    |    8.072     |
            
	  	  
* MDX Leaderboard B (bleeding)
    * Submission repo: [MDX2023-bleeding-submission](https://gitlab.aicrowd.com/subatomicseer/MDX2023-bleeding-submission)
	* Submission ID: 220344
	* Submitter: subatomicseer
	* Final rank: 3rd place
	* Final scores:
  
	  |  SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
	  |:--------:| :------: | :-------: | :-------: | :--------: |
	  |   6.314	|  6.331	  |  6.864	  |  4.591	  |  7.469   |

* MDX Leaderboard C (Open)
    * Submission repo: [MDX2023-external-data-submission](https://gitlab.aicrowd.com/subatomicseer/mdx2023-external-data-submission)
    * Submission ID: 220008
    * Submitter: subatomicseer
    * Final rank: 8th place
    * Final scores:
  
      |  SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
      |:--------:| :------: | :-------: | :-------: | :--------: |
      |   8.537	|  9.328	  |  9.328	  |  6.182	  |  9.311   |

* CDX Leaderboard A and B
    * Submission repo: [CDX2023-dnr-submission](https://gitlab.aicrowd.com/subatomicseer/CDX2023-dnr-submission)
    * Submission ID: 220293
    * Submitter: subatomicseer
    * Final rank: 3rd place (A), 6th place (B)
    * Final scores:
  
      | SDR_mean  | SDR_dialog | SDR_effect | SDR_music  |
      |:--------:| :------: | :-------: | :-------: | 
      | 4.144	|7.178	|3.466	|2.011	  | 

## Model Descriptions
Throughout the challenge, we used the following models:
- DWT-Transformer-UNet: [DWT-Transformer-UNet.md](docs%2Fmodel_descriptions%2FDWT-Transformer-UNet.md) 
- Wavelet-HTDemucs: [Wavelet-HTDemucs.md](docs%2Fmodel_descriptions%2FWavelet-HTDemucs.md)
- BSRNN: https://arxiv.org/abs/2209.15174

## Noise Robust Training for MDX Leaderboard A and B
The description of noise robust training losses is provided in:
- [Noise Robust Training](docs/model_descriptions/NOISE_ROBUST_TRAINING.md)

## Data Augmentations
The description of data augmentations is provided in:
- [Data Augmentations](docs/model_descriptions/DATA_AUGMENTATIONS.md)

## Environment setup

```shell
conda create -n sdx2023 python=3.8
conda activate sdx2023
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

## Training instructions for individual models are provided in the `docs` folder:

Note regarding the datasets
All datasets are assumed to be in a directory named `DATASETS` in the root directory of this project as shown below:
- `DATASETS/SDX2023/moisesdb23_bleeding_v1.0`
- `DATASETS/SDX2023/moisesdb23_labelnoise_v1.0`
- `DATASETS/DnR/dnr_v2`
- `DATASETS/MUSDB-HQ`

### MDX track
Leaderboard A (labelnoise): [docs/TRAINING(MDX-Labelnoise).md](docs/TRAINING(MDX-Labelnoise).md)

Leaderboard B (bleeding): [docs/TRAINING(MDX-Bleeding).md](docs/TRAINING(MDX-Bleeding).md)

Leaderboard C (Open): [docs/TRAINING(MDX-Open).md](docs/TRAINING(MDX-Open).md)

### CDX track
Leaderboard A and B: [docs/TRAINING(CDX-DnR).md](docs/TRAINING(CDX-DnR).md)




Either copy the datasets to the above locations, or create symbolic links to the datasets, 
or you can change the dataset paths in the config files and pre-processing scripts.


## References

[1] S. Rouard, et al., "Hybrid Transformers for Music Source Separation", Arxiv 2022

[2] Y. Luo, et al., "Music Source Separation with Band-split RNN", Arxiv 2022

[3] S. Uhlich, et al., "Improving music source separation based on deep neural networks through data augmentation and network blending", ICASSP 2017.

[4] S. Wisdom, et al., "Unsupervised Sound Separation Using Mixture Invariant Training", NeurIPS 2020

[5] A. Tarvainen, et al., "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results", NIPS 2017

[6] T. Ishida, et al., "Do We Need Zero Training Loss After Achieving Zero Training Error?", ICML 2020

[7] T. Nakamura, et al., "Time-Domain Audio Source Separation Based on Wave-U-Net Combined with Discrete Wavelet Transform", ICASSP 2020
