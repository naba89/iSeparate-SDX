# Leaderboard A and B (Used only DnR dataset)

**Final LB Rank : 3rd (A), 6th (B)**

**Final LB Score: 4.144 dB**


The final submission is a weighted blending of two models. Training details for each model is given below.

**NOTE**: All models were trained on 4x 80GB A100 GPUs. 


## Step 1: Data pre-processing
The `tr-subset` of the DnR dataset was pre-processed to remove silence from the music and speech stems.

Using the following command:
```shell
python utitlity_scripts/pre_process_dnr.py
```

For validation, we use 32 mixtures from the `cv-subset`. You can prepare the validation data using the following command:
```shell
python utitlity_scripts/create_sdx_valid.py
```

## Step 2: Train DWT-Transformer-UNet model
Memory requirements:
    ~66GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/dwt_transformer_unet/cdx/config_cdx_dwt.yaml
```

## Step 3: Train BSRNN model
Memory requirements:
    ~37GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/cdx/config_cdx_music.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/cdx/config_cdx_sfx.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/cdx/config_cdx_speech.yaml
```


## Step 4: Ensemble

The final submission is a weighted blending of the three models. 
The weights were determined based on the LB performance of individual models.

The weights are as follows:

| Source | BSRNN | DWT-Transformer-UNet |
|--------|-------|----------------------|
| music  | 0.18  | 0.82                 |
| speech | 0.18  | 0.82                 |
| sfx    | 0.82  | 0.18                 |