# NeurIPS'25: Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data

This repository is PyTorch implementation for **PEPSY**, a novel federated multimodal learning that explores **missing patterns** across clients, accepted at NeurIPS-25: [Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](https://neurips.cc/virtual/2025/poster/115784).

![pepsy](./assets/pepsy_overview.png)

## Overview

* **Problem.** One of main challenges in federated multimodal learning is when each client observes a different subset of modalities and and might also have missing input features within each modality. 

* **Key Ideas.** To tackle these challenges, **PEPSY** embeds these different modality combinations (or __missing patterns__) of each client into a so-called __embedding controls__, which later serve as reconfiguration signals to reconfigure the data-missing bias features extracted from the raw data. Furthermore, the embedding controls can be aggregated across clients with similar missing patterns to enhance the robustness of reconfiguration signals in adapting the global representation.

* **Empirical Results.** Empirical results on multiple federated multimodal benchmarks with diverse data-missing patterns across clients demonstrate the efficacy of the PEPSY, achieving up to 36.45% performance improvement under severe data incompleteness. 

* **Theoretical Analysis.** PEPSY is supported by a theoretical analysis with an explicit performance bound that matches our empirical observations.

## To-do

We will update the repository if necessary. Overall, all the models and training strategies are available.

+ [&check;] PEPSY and baselines

+ [&check;] Dataloader, client and servers
+ [&check;] Training scripts
+ [&cross;] Data preprocessing
+ [&cross;] Data samples

## Project Structure

```
├── algorithm/                  # Server Aggregation Components
├── benchmark/                  # Clients' Model Architectures
├── scripts/                    # Experiment scripts
├── utils/                      # Utility files
├── DATA.md                     # Data readme files
├── generate_fedtask.py         # Create data files (client data partition) before running FL
├── main.py                     # Main running files
├── README.md                   # This file
└──requirements.txt            # Required Python packages
```

## Requirements
```
conda create -n pepsy python=3.9 pip
conda activate pepsy
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Prepare the multimodal datasets with simulated missing data patterns as described in the paper (e.g., PTBXL, Sleep-EDF). Note that different missing simulation patterns are algorithmatically handled instead of manual drop-out, hence only preprocessed data is needed.

We use two multimodal datasets: [PTBXL](https://physionet.org/content/ptb-xl/1.0.1/) (12 modalities) and [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) (5 modalities). Please refer to our paper for more details.

### Training and Evaluation

**Quick start**

Note that the following script runs multiple experiments at the same time. You should adjust the script before running to avoid any failure.

```
cd src
bash ./script/edf_classification_lm/non-iid/PEPSY/NP_FedAvg/PEPSY_pm02.sh
```

**Detailed Running**

* Step 1: Generate federated task with custom skewness (IID or Non-IID):
```
python generate_fedtask.py \
    --benchmark edf_classification_lm \
    --dist 1 \
    --skew 0.5 \
    --num_clients 32 \
    --seed 0 
```

* Step 2: Evaluate algorithms on genereted tasks:

```
python main.py \
    --task edf_classification_lm_cnum32_dist1_skew0.5_seed0_full_modal_local_missing \
    --model pepsy \
    --algorithm multimodal.edf_classification_lm.pepsy.probabilistic_prompt_1enc_12enc_emb \
    --sample uniform \
    --aggregate other \
    --num_rounds 500 \
    --proportion 0.33 \
    --num_epochs 3 \
    --learning_rate 0.1 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 1 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --ps 0.6 \
    --pm 1.0 \
    --graph_quantile 0.0 \
    --contrastive_weight 0.2 \
    --num_outer_loops 5 \
    --num_fedavg_rounds 0 \
    --wandb
```

### Empirical Results

Empirical results on multiple federated multimodal benchmarks with diverse data-missing patterns across clients demonstrate the efficacy of the PEPSY, achieving up to 36.45% performance improvement under severe data incompleteness.

![results](./assets/results.png)

## Bibtex

If you use our paper and source code or found them helpful, please considering citing:
```
@inproceedings{nguyen2025pepsy,
  title = {Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data},
  author = {Duong M. Nguyen and Trong Nghia Hoang and Thanh Trung Huynh and Quoc Viet Hung Nguyen and Phi Le Nguyen},
  booktitle = {The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2025},
}

@inproceedings{nguyen2025fedmac,
  title={FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization}, 
  author={Nguyen, Manh Duong and Nguyen, Trung Thanh and Pham, Huy Hieu and Hoang, Trong Nghia and Nguyen, Phi Le and Huynh, Thanh Trung},
  booktitle={2024 22nd International Symposium on Network Computing and Applications (NCA)}, 
  year={2024}
}
```

## Acknowledgement

The implementation of this repository is based on the following open-source projects:
* [easyFL: A lightning Framework for Federated Learning](https://github.com/WwZzz/easyFL/tree/main)