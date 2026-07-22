# FearConditioningAI

> **A deep neural network model of associative emotional (Pavlovian fear) learning**

[![arXiv](https://img.shields.io/badge/arXiv-2607.19327-b31b1b)](https://arxiv.org/abs/2607.19327)
[![Neural Computation](https://img.shields.io/badge/Neural%20Computation-in%20press-green)](https://arxiv.org/abs/2607.19327)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

This project develops the **Visual-Valence Model**, a deep-learning computational model of associative emotional (Pavlovian) learning. The model has three components (see Fig. 1 of the paper):

- **Visual Cortex module** — VGG-16 (Simonyan & Zisserman, 2015), pretrained on ImageNet (Deng et al., 2009) and held frozen, standing in for the ventral visual stream.
- **Shortcut Pathway** — a parallel route from the Visual Cortex module's early layers, pooled at multiple scales and combined via an Efficient Channel Attention (ECA) mechanism (Wang et al., 2020), modeling proposed rapid cortical/subcortical routes from early vision to emotion-related circuitry.
- **Valence Module** — two fully connected layers plus a one-unit output, inspired by the lateral (LA) and central (CE) nuclei of the amygdala and the orbitofrontal cortex (OFC), which combines the Visual Cortex and Shortcut Pathway outputs into a single scalar valence prediction (1 = extreme displeasure, 9 = extreme pleasure).

In code, this architecture is implemented as `Visual_Cortex_Amygdala` in `models/VGG_Model.py` (`--model_to_run 6`, the model used throughout the study).

Training proceeds in stages: the model first learns to predict image valence from natural scenes (first the Cowen & Keltner, 2017 Videoframe dataset, then the International Affective Picture System, IAPS), is fine-tuned to a quadrant-based input layout, and is finally run through a novel Pavlovian conditioning paradigm that pairs a conditioned stimulus (a Gabor patch, CS+) with an unconditioned stimulus (an IAPS image, US) of known valence (see [Training](#training) for the full pipeline). After conditioning, the model reproduces several hallmarks of human associative learning — association formation and generalization — and its conditioned/unconditioned stimulus representations become increasingly aligned, both at the single-unit and population level, consistent with empirical human data.

## Publications

**Please cite our work if you use this code**

| Type | Title | Venue | Status |
|------|-------|-------|--------|
| Journal | Associative Emotional Learning in Convolutional Neural Networks | Neural Computation | Accepted / in press |

**Authors:** Seowung Leem, Andreas Keil, Mingzhou Ding, Ruogu Fang

- arXiv: https://arxiv.org/abs/2607.19327

---

## Repository Layout

```
FearConditioningAI/
├── environment-linux.yml          # Conda env: Linux + NVIDIA GPU (CUDA 12.1)
├── environment-mac.yml            # Conda env: macOS (CPU / Apple Silicon MPS)
├── environment-windows.yml        # Conda env: Windows + NVIDIA GPU (CUDA 12.1)
├── requirements.txt                # Legacy full conda export (py36), kept for reference
├── Gabor4Seowung.m                 # MATLAB script that generates the Gabor-patch CS stimuli
├── dataloader.py                   # Dataset / DataLoader construction (regression, quadrant, conditioning)
├── utils.py                        # Eval loops, checkpoint I/O, image transforms, plotting/analysis helpers
│
├── models/
│   ├── network.py                  # Base VGG-16 backbone + earlier "Amygdala" model iterations
│   ├── VGG_Model.py                 # Visual_Cortex_Amygdala — the Visual-Valence Model (Visual Cortex / Shortcut Pathway / Valence Module)
│   ├── VGG_Model_Conditioning.py    # Single-pathway VGG-16 + ECA attention variants
│   ├── VGG_classification.py        # Amy_IntermediateRoad (LA/CE nuclei) + plain VGG classifier
│   ├── eca_module.py                 # Efficient Channel Attention (ECA) module
│   └── filter_module.py              # Gaussian smoothing / Gaussian noise transforms
│
├── torchsampler/                    # Vendored ImbalancedDatasetSampler (ufoym, MIT license)
│
├── train_regression.py              # Stages 0-1: pretrain on Videoframe, then fine-tune on full-size IAPS images
├── train_finetuning.py              # Stage 2: fine-tune to the quadrant input layout
├── train_conditioning2.py            # Stage 3: Pavlovian conditioning (CS Gabor patch × US IAPS image)
│
├── test_gaborpatch_iteration.py     # Track one Gabor patch's decoded valence across checkpoints/epochs
├── test_gaborpatches.py              # Compare decoded valence across many Gabor patches, pre- vs. post-conditioning
├── test_result_concatenation.py     # Merge repeated test runs into learning-curve CSVs
│
├── Channel_Activity_Extraction.py    # Dump per-image layer activations for downstream analysis
├── Manifold_Visualization.py         # t-SNE visualization of extracted activations
├── SVM_Analysis_Emotion.py           # SVM valence decoding, trained on US and tested on US/CS
├── SVM_Analysis_Before_After.py      # SVM generalization: pre- vs. post-conditioning feature space
│
├── data/                              # (not tracked) datasets — see Dataset section
├── savedmodel/                        # (not tracked) trained model checkpoints
└── result/                            # (not tracked) analysis outputs (CSVs, figures)
```

---

## Hardware Requirements

| Platform | Configuration |
|----------|---------------|
| Linux | NVIDIA GPU, CUDA 12.1 (see `environment-linux.yml`) |
| Windows | NVIDIA GPU, CUDA 12.1 (see `environment-windows.yml`) |
| macOS | CPU, or Apple Silicon via the MPS backend (see `environment-mac.yml`) |

> The training/analysis scripts pin a specific GPU index via `os.environ["CUDA_VISIBLE_DEVICES"]` near the top of each file (e.g. `"0"`, `"2"`, `"3"`). Edit this to match a device available on your machine, or set it to `"-1"`/remove it to fall back to CPU.

---

## Installation

```bash
# Linux + NVIDIA GPU
conda env create -f environment-linux.yml
# Windows + NVIDIA GPU
conda env create -f environment-windows.yml
# macOS (CPU / Apple Silicon)
conda env create -f environment-mac.yml

conda activate fear-conditioning-ai
```

`requirements.txt` is a full conda-environment export from an earlier (Python 3.6) setup and is kept only for historical reference — prefer the `environment-*.yml` files above for a fresh install.

---

## Dataset

Training uses two stimulus types:

- **Unconditioned stimulus (US):** natural scene images from the **International Affective Picture System (IAPS)**, each labeled with a human valence rating (1–9 scale). *Because of a data-use/confidentiality agreement, the IAPS images themselves are not redistributed in this repository (see `.gitignore`); you must obtain access to IAPS independently.*
- **Conditioned stimulus (CS):** Gabor patches varying in orientation, spatial frequency, and contrast, generated via `Gabor4Seowung.m`.

All data is expected under `./data`, referenced by the training/analysis scripts' `--data_dir`, `--TRAIN`/`--VAL`/`--TEST` (or `--gabor_dir`) folder-name arguments, e.g.:

```
data/
├── IAPS_10-10-80_train3/           # US training images
├── IAPS_10-10-80_train3.csv        # image, extension, sd, valence (no header)
├── IAPS_10-10-80_val3/
├── IAPS_10-10-80_val3.csv
├── IAPS_10-10-80_test3/
├── IAPS_10-10-80_test3.csv
├── IAPS_Conditioning_Unpleasant2/  # US images paired with the unpleasant CS during conditioning
├── IAPS_Conditioning_Unpleasant2.csv
├── IAPS_Conditioning_Pleasant2/    # US images paired with the pleasant CS during conditioning
├── IAPS_Conditioning_Pleasant2.csv
└── gabor_patch_full/               # CS Gabor patches, e.g. gabor-gaussian-45-freq20-cont50.png
```

Each label CSV has no header and four columns: image name, file extension, an unused/SD column, and the valence label.

---

## Training

The Visual-Valence Model architecture (`--model_to_run 6`, the default) is fixed throughout the study; training instead proceeds through the stages below, each stage's `--file_name` checkpoint feeding into the next.

### Stage 0 — pretrain on the Cowen & Keltner Videoframe dataset (`train_regression.py`)

Before ever seeing IAPS, the paper first trains the Visual-Valence Model from scratch to decode valence from the Cowen & Keltner (2017) Videoframe dataset (2,185 emotion-eliciting videos, one static frame sampled every 10th frame, split 8:1:1 train/val/test). This dataset is not distributed with this repository; point `--TRAIN`/`--VAL`/`--TEST`/`--csv_*` at your own Videoframe-style split (image, extension, unused, valence columns) to reproduce it:

```bash
python train_regression.py \
  --data_dir ./data --TRAIN videoframe_train --VAL videoframe_val --TEST videoframe_test \
  --csv_train ./data/videoframe_train.csv --csv_val ./data/videoframe_val.csv --csv_test ./data/videoframe_test.csv \
  --batch_size 128 --lr 2e-5 --epoch 50 --is_fine_tune False \
  --model_dir ./savedmodel/ --model_name vca_ckvideo_batch128_2e-5_epoch20.pth
```

### Stage 1 — fine-tune on full-size IAPS images (`train_regression.py`)

Re-run `train_regression.py`, this time loading the Stage 0 checkpoint (`--file_name`) and continuing training on full-size IAPS images, to adapt the model to the US stimuli used in conditioning.

```bash
python train_regression.py \
  --data_dir ./data \
  --TRAIN IAPS_10-10-80_train3 --VAL IAPS_10-10-80_val3 --TEST IAPS_10-10-80_test3 \
  --csv_train ./data/IAPS_10-10-80_train3.csv --csv_val ./data/IAPS_10-10-80_val3.csv --csv_test ./data/IAPS_10-10-80_test3.csv \
  --batch_size 10 --lr 2e-4 --epoch 100 \
  --model_dir ./savedmodel/ --model_name vca_IAPS_batch10_lr2e-4_epoch100.pth \
  --is_fine_tune True --file_name vca_ckvideo_batch128_2e-5_epoch20.pth
```

### Stage 2 — `train_finetuning.py`

Fine-tunes the Stage 1 checkpoint (`--file_name`) into the quadrant-cropped input layout later used for conditioning (the US confined to the 4th quadrant, matching where it will appear during Pavlovian conditioning).

```bash
python train_finetuning.py \
  --data_dir ./data \
  --TRAIN IAPS_10-10-80_train3 --VAL IAPS_10-10-80_val3 --TEST IAPS_10-10-80_test3 \
  --csv_train ./data/IAPS_10-10-80_train3.csv --csv_val ./data/IAPS_10-10-80_val3.csv --csv_test ./data/IAPS_10-10-80_test3.csv \
  --batch_size 16 --lr 1e-5 --epoch 100 \
  --model_dir ./savedmodel --model_name vca_IAPS_quadrant_batch16_lr1e-5.pth \
  --is_fine_tune True --file_name vca_IAPS_batch10_lr2e-4_epoch23.pth
```

### Stage 3 — `train_conditioning2.py`

Runs the Pavlovian conditioning paradigm: on every trial, a CS+ Gabor patch (`--gabor_dir1`/`--gabor_dir2`, placed in the 2nd quadrant) is paired with an unpleasant/pleasant US image (`--TRAIN`/`--TRAIN2`, placed in the 4th quadrant), fine-tuning the Stage 2 checkpoint (`--file_name`). The 45° Gabor is always paired with unpleasant US images and the 135° Gabor with pleasant US images.

```bash
python train_conditioning2.py \
  --data_dir ./data \
  --gabor_dir1 ./data/gabor_patch_full/freq20/gabor-gaussian-45-freq20-cont50.png \
  --gabor_dir2 ./data/gabor_patch_full/freq20/gabor-gaussian-135-freq20-cont50.png \
  --TRAIN IAPS_Conditioning_Unpleasant2 --TRAIN2 IAPS_Conditioning_Pleasant2 \
  --batch_size 10 --lr 1e-4 --epoch 100 \
  --model_dir ./savedmodel/ --model_name base_model_conditioned_orientation.pth \
  --is_fine_tune True --file_name base_model_vca_IAPS_quadrant.pth
```

`--model_save_mode` controls checkpointing: `1` saves every epoch, `2` saves only when validation loss improves, `3` (default) saves just the first and last epoch.

---

## Testing & Analysis

Run after a conditioned checkpoint (Stage 3) is available in `savedmodel/`.

```bash
# Track one Gabor patch's decoded valence across every checkpoint in savedmodel/
python test_gaborpatch_iteration.py --gabor_patch gabor-gaussian-45-freq20-cont50.png

# Compare decoded valence across many Gabor patches, before vs. after conditioning
python test_gaborpatches.py --initial_model_name base_model_vca_IAPS_quadrant.pth \
  --conditioned_model_name base_model_conditioned_orientation_epoch100.pth

# Merge repeated runs of test_gaborpatch_iteration.py into learning-curve CSVs
python test_result_concatenation.py --result_dir ./result/
```

For representation-level analysis (feature extraction → t-SNE / SVM):

```bash
# 1. Extract per-image layer activations (repeat for --image_to_extract image and gabor)
python Channel_Activity_Extraction.py --image_to_extract gabor --module_to_extract VCA_FC

# 2a. t-SNE manifold visualization of the extracted activations
python Manifold_Visualization.py --model_name base_model_conditioned_orientation_epoch100

# 2b. SVM valence decoding: trained on US, tested on held-out US or on CS Gabor orientations
python SVM_Analysis_Emotion.py --what_to_analyze US   # or: --what_to_analyze CS

# 2c. SVM generalization: does a pre-conditioning classifier still work post-conditioning?
python SVM_Analysis_Before_After.py \
  --initial_model base_model_vca_IAPS_quadrant --conditioned_model base_model_conditioned_orientation_epoch100
```

---

## Citations

If you use this code or its datasets, please cite our paper as well as the underlying datasets and methods it builds on:

**This work:**
> Leem, S., Keil, A., Ding, M., & Fang, R. (2026). Associative Emotional Learning in Convolutional Neural Networks. *Neural Computation* (in press). https://arxiv.org/abs/2607.19327

**Datasets:**
> Bradley, M. M., & Lang, P. J. (2007). The International Affective Picture System (IAPS) in the study of emotion and attention. In *Handbook of Emotion Elicitation and Assessment* (pp. 29–46). Oxford University Press.

> Cowen, A. S., & Keltner, D. (2017). Self-report captures 27 distinct categories of emotion bridged by continuous gradients. *Proceedings of the National Academy of Sciences*, 114(38), E7900–E7909. https://doi.org/10.1073/pnas.1702247114

> Bradley, M. M., & Lang, P. J. (1994). Measuring emotion: The self-assessment manikin and the semantic differential. *Journal of Behavior Therapy and Experimental Psychiatry*, 25(1), 49–59. https://doi.org/10.1016/0005-7916(94)90063-9

**Model components:**
> Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*. https://doi.org/10.48550/arXiv.1409.1556

> Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248–255. https://doi.org/10.1109/CVPR.2009.5206848

> Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11531–11539. https://doi.org/10.1109/CVPR42600.2020.01155

**Conditioning paradigm background:**
> Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In *Classical Conditioning II: Current Research and Theory* (pp. 64–99). Appleton-Century-Crofts.

---

## Contact

| Name | Email |
|------|-------|
| Seowung Leem | leem.s@ufl.edu |
| Dr. Ruogu Fang | ruogu.fang@bme.ufl.edu |

## License

This project is licensed under the **MIT License** 
