# Leaderboard B (Bleeding dataset only)

**Final LB Rank : 3rd**

**Final LB Score: 6.314 dB**

The final submission is a weighted blending of two models. Training details for each model is given below.

**NOTE**: All models were trained on 4x 80GB A100 GPUs. 

**NOTE**: No validation split is used, 
training was stopped based on one or more of the following reasons:
- training loss instability 
- training loss saturation
- leaderboard score not improving 
- GPU unavailability 
- lack of time towards the end of competition.

## Step 1: Train Wavelet-HTDemucs model
Memory requirements:
    ~28GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/wavelet_htdemucs/config_mdx_bleeding.yaml
```

## Step 2: Train DWT-Transformer-UNet model
Memory requirements:
    ~36GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/dwt_transformer_unet/mdx/config_mdx_dwt_bleeding.yaml
```


## Step 3: Ensemble

The final submission is a weighted blending of the two models. 
The weights were determined based on the LB performance of individual models.

The weights are as follows:

| Source | DWT-Transformer-UNet | Wavelet-HTDemucs |
|--------|----------------------|------------------|
| bass   | 0.5                  | 0.5              |
| drums  | 0.6                  | 0.4              |
| other  | 0.4                  | 0.6              |
| vocals | 0.4                  | 0.6              |