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
- Refer to this github [repo](https://github.com/cleinc/bts/tree/master), download the files in `bts/utils`, `extract_official_train_test_set_from_mat.py` and `splits.mat`
- Put all downloaded files in one folder and run
  ```
  python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat dataset/nyu_depth_v2/official_splits/
  ```

**KITTI**
- Download the ground truth labels from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip).
- Download the RGB samples through this [script](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip).
- Put all downloaded files in one folder and make sure the path is similar with the one listed in the `data_split/kitti_depth/eigen_test_files_with_gt.txt`
