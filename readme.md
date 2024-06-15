# PaPa: Propagation pAttern Enhanced Prompt leArning for Zero-shot Rumor Detection

## Overview
"PaPa: Propagation Pattern Enhanced Prompt Learning for Zero-shot Rumor Detection" by Heng Zhang, Peng Wu, and Li Pan is a research work focused on enhancing zero-shot rumor detection capabilities in social media contexts through the incorporation of propagation patterns.

## Datasets
Our study leverages several real-world datasets, which are preprocessed for your convenience. They can be accessed as follows:

- **Preprocessed Datasets**: Available on [Baidu Netdisk](https://pan.baidu.com/s/15ERoMXhVwu9NNHJ1BnrsAQ). Use the extraction code **0525** to download.

- **Raw Datasets**:
  - Twitter15 and Twitter16: [Dropbox Link](https://www.dropbox.com/sh/w3bh1crt6estijo/AAD9p5m5DceM0z63JOzFV7fxa?dl=0)
  - Terrorist and Gossip: [Figshare - PHEME Dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
  - Twitter and Weibo: [Dropbox Rumdect Zip](https://www.dropbox.com/scl/fi/t8irh7mg2asjcameqzoyz/rumdect.zip?rlkey=ngpfapo8ftr7zxqohbj7onqm3&e=1&dl=0)
  - Twitter-COVID19 and Weibo-COVID19: [GitHub - ACLR4RUMOR-NAACL2022](https://github.com/DanielLin97/ACLR4RUMOR-NAACL2022)

## Dependencies
To replicate our experimental setup, please install the following dependencies with their respective versions:

- Python: `3.8.10`
- NumPy: `1.23.5`
- PyTorch: `2.2.2`
- Transformers: `4.38.1`
- tqdm: `4.66.1`

## Usage
To train and test the PaPa model, follow these steps:

1. Prepare the datasets in the `./data` directory.
2. Ensure pretrained parameters and configuration files for `xlm-roberta-base` are located in `../xlm-roberta-base`.
3. Execute the training script with:
   ```bash
   python train.py