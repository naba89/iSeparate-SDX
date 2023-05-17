# Leaderboard C (Open)

**Final LB Rank : 8th**

**Final LB Score: 8.537 dB**


The final submission is a weighted blending of three models. Training details for each model is given below.

**NOTE**: All models were trained on 4x 80GB A100 GPUs. 

## Datasets used:
Training: total 347 songs
Public datasets:
- MUSDB18-HQ Train (84)
- MUSDB18-HQ Test (50)
- MedleyDB (songs not already included in MUSDB18-HQ) (74)

Validation: total 16 songs
- MUSDB18-HQ Validation (14)
- MedleyDB (2)

## Step 1: Train DWT-Transformer-UNet model
Memory requirements:
    ~66GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/dwt_transformer_unet/mdx/config_mdx_dwt.yaml
```

## Step 2: Train BSRNN model
Memory requirements:
    ~47GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_md_labelnoise_bass.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_md_labelnoise_drums.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_md_labelnoise_other.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_md_labelnoise_vocals.yaml
```

## Step 3: Use HTDemucs model released by the demucs library
https://github.com/facebookresearch/demucs

Trained on 800 songs

Inference settings:
- shifts: 1
- overlap: 0.25


## Step 4: Ensemble

The final submission is a weighted blending of the three models. 
The weights were determined by grid search on the 16 validation songs.

The weights are as follows:

| Source | BSRNN | DWT-Transformer-UNet | HTDemucs |
|--------|-------|----------------------|----------|
| bass   | 0     | 0.3                  | 0.7      |
| drums  | 0.3   | 0.2                  | 0.5      |
| other  | 0     | 0.4                  | 0.6      |
| vocals | 0.4   | 0.2                  | 0.4      |