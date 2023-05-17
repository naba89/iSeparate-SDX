# Leaderboard A (LabelNoise dataset only)

**Final LB Rank : 2nd**

**Final LB Score: 6.601 dB**

The final submission is a weighted blending of three models. Training details for each model is given below.

**NOTE**: All models were trained on 4x 80GB A100 GPUs. 

**NOTE**: No validation split is used, 
training was stopped based on one or more of the following reasons:
- training loss instability 
- training loss saturation
- leaderboard score not improving
- GPU unavailability 
- lack of time towards the end of competition.

The config files contains the number of epochs trained for each model.

## Step 1: Train Wavelet-HTDemucs model
Memory requirements:
    ~28GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/wavelet_htdemucs/config_mdx_labelnoise.yaml
```

## Step 2: Train DWT-Transformer-UNet model
Memory requirements:
    ~36GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/dwt_transformer_unet/mdx/config_mdx_dwt_labelnoise.yaml
```

## Step 3: Data Cleaning

Below is a description of the data cleaning process. If you want to skip the details and
just want to create the clean dataset, run the following command:
```shell
python utility_scripts/create_labelnoise_clean_datasets_from_filelists.py
```

### Step 3.1: Manual selection of clean samples from the labelnoise dataset:
Initially I manually checked about 30 songs from the labelnoise dataset and extracted the clean stems.
The list of songs and stems is present in `file_lists/labelnoise_clean_v2.txt`

This version of the clean dataset was used to train the BSRNN model for the `other` stem

Later, I used an automatic way to find clean samples from the labelnoise dataset, as described below.

Most of the songs in the manual version are also present in the automatic version, except for a few songs.

### Step 3.2: Use DWT-Transformer-UNet model to find and filter clean samples from the labelnoise dataset
This version of the clean dataset was used to train the BSRNN model for the `bass, drums, vocals` stems

Run the following command to evaluate the DWT-Transformer-UNet model on the labelnoise dataset.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh evaluate_dist.py configs/dwt_transformer_unet/mdx/config_eval_labelnoise.yaml
```

This will produce a csv file with SDRs for all stems for all songs in the labelnoise dataset. 
You can find the generated csv file in `file_lists/moisesdb23_labelnoise_v1.0_sep_dwt_scores.csv`

The idea is that, if the model is able to separate well and 
  the stem in the dataset is clean, then the SDRs should be high.

After that run the following script to filter out the clean stems from the labelnoise dataset, with a threshold of 9dB SDR.
```shell
python utility_scripts/create_labelnoise_clean_dataset.py
```
This will create a new dataset with only cleans stems. Not all stems might be present for all songs.

**Following this, I manually verified the stems with low SDRs between 9~12 dB, and removed some obvious noisy stems**

The final list of songs and stems is present in `file_lists/dwt_labelnoise_clean_v2.txt`

## Step 4: Train BSRNN model
Memory requirements:
    ~37GB per GPU 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_mdx_labelnoise_bass.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_mdx_labelnoise_drums.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_mdx_labelnoise_other.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash launcher.sh train_dist.py configs/bsrnn/mdx_labelnoise/config_mdx_labelnoise_vocals.yaml
```

## Step 5: Ensemble

The final submission is a weighted blending of the three models. 
The weights were determined based on the LB performance of individual models.

The weights are as follows:

| Source | BSRNN | DWT-Transformer-UNet | Wavelet-HTDemucs |
|--------|-------|----------------------|------------------|
| bass   | 0.1   | 0.2                  | 0.7              |
| drums  | 0.1   | 0.8                  | 0.1              |
| other  | 0.3   | 0.3                  | 0.4              |
| vocals | 0.7   | 0.1                  | 0.2              |