# Marigold Reproducibility by xyclone10

This reproducibility project examines [Marigold](https://ieeexplore.ieee.org/abstract/document/10655342/), a state-of-the-art monocular depth estimation model. We provide a concise, step-by-step process to replicate the results exactly as reported in the original paper.

## Setup
The code was tested on:
- Ubuntu 22.04.5 LTS
- Python 3.10.12
- CUDA 11.5
- NVIDIA RTX 6000 Ada Generation

### Clone This Repository
```
git clone https://github.com/xyclone10/repro-marigold.git
```

### Install The Dependencies
```
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

### Prepare Datasets
There are four datasets in total, [Hypersim](https://github.com/apple/ml-hypersim) and [VKITTI2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) for training, [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [KITTI](https://www.cvlibs.net/datasets/kitti/) for testing.

**Hypersim**
- Download the dataset from this [script](https://github.com/apple/ml-hypersim/blob/20f398f4387aeca73175494d6a2568f37f372150/code/python/tools/dataset_download_images.py).
- Download the scene split file from [here](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv).
- Run the preprocessing script:
  ```
  python script/dataset_preprocess_depth/hypersim/preprocess_hypersim.py --split_csv /path/to/metadata_images_split_scene_v1.csv
  ```

**VKITTI2**
- Download the RGB samples from [here](https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar).
- Download the depth labels from [here](https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar).
- Extract all files.

**NYUv2**
- Download the files through
  ```
  wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
  ```
- Refer to this GitHub [repo](https://github.com/cleinc/bts/tree/master), download the files in `bts/utils`, `extract_official_train_test_set_from_mat.py`, and `splits.mat`
- Put all downloaded files in one folder and run
  ```
  python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat dataset/nyu_depth_v2/official_splits/
  ```
- Check the file path and make sure it is similar to the one listed in the `data_split/nyu_depth/labeled/filelist_test.txt`.

**KITTI**
- Download the ground truth labels from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip).
- Download the RGB samples through this [script](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip).
- Put all downloaded files in one folder and make sure the path is similar to the one listed in the `data_split/kitti_depth/eigen_test_files_with_gt.txt`

Put all prepared datasets under the path of `${BASE_DATA_DIR}`.

## Reproduce The Experiments

### Training
Set environment parameters for the data directory:
```
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`.
Alternative [link](https://www.kaggle.com/models/stabilityai/stable-diffusion-v2).

Run the training script
```
python script/depth/train.py --config config/train_marigold_depth.yaml
```

### Evaluation and Inference
Set environment parameters for the data directory:
```
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Run the inference script first
```
bash script/depth/eval_old/11_infer_nyu.sh
```

Then, after that, run the evaluation script
```
bash script/depth/eval_old/12_eval_nyu.sh
```

## Experiments Guidebook

### Zero-shot Performance
To reproduce the best result as stated in the paper, do not change the config, and just run the experiment as it is.

### Ablation: Training Noise
In the file `config/train_marigold_depth.yaml`, change the `multi_res_noise strength` for multi-resolution noise and `multi_res_noise annealed` for the annealed schedule.

### Ablation: Training Data Domain
In the file `config/dataset_depth/dataset_train.yaml`, change the `prob_ls` configuration; the first index is for Hypersim, the second index is for VKITTI2.

### Ablation: Test-time Ensembling
In the file `script/depth/eval_old/11_infer_nyu.sh`, change the `ensemble_size`.

### Ablation: Number of Denoising Steps
In the file `script/depth/eval_old/11_infer_nyu.sh`, change the `denoise_steps`.
