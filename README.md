# nanoJiT

This is a minimal, faithful PyTorch implementation of the "Just image Transformers" architecture described in the paper ["Back to Basics: Let Denoising Generative Models Denoise"](https://www.arxiv.org/abs/2511.13720) (Li et al., 2026). It implements the JiT-B/16 architecture but adapts the training loop for the ImageNette dataset to allow training on basic GPUs.

## about the paper

<img width="360" height="645" alt="jit for diffusion diagram" src="https://github.com/user-attachments/assets/285017b3-e3e6-4473-8a90-2cbf87d55b83" />


The paper argues that modern diffusion models rely too heavily on complex tricks like VAEs, tokenizers, and latent spaces because they are trying to predict noise (epsilon). Predicting random noise is difficult in high-dimensional spaces like raw pixels.

The authors propose "x-prediction," where the network directly predicts the clean image. This leverages the "Manifold Assumption," which states that natural images lie on a low-dimensional manifold. By predicting the clean image directly, simple Transformers can function effectively on raw pixel patches without needing tokenizers or pre-training.

## architecture and implementation features

This repository implements the "Advanced" JiT-B/16 configuration described in Section 4.4 of the paper. It is not a simplified toy model; it contains the exact architectural components used in the study:

* **Core Architecture:** Vision Transformer (ViT) with a hidden dimension of 768, 12 heads, and 12 layers.
* **Patching:** Operates on raw 16x16 pixel patches.
* **Modern Components:** Includes SwiGLU activations, RMSNorm, Rotary Positional Embeddings (RoPE), and QK-Norm.
* **Conditioning:** AdaLN-Zero for time/class conditioning and In-Context Conditioning strategy.
* **Objective:** Rectified Flow with velocity matching loss (v-loss), but the network output is parameterized to predict the clean image (x).
* **Time Sampling:** Logit-Normal distribution for sampling training timesteps.

## training details and hardware

This implementation is tuned for the ImageNette dataset (a 10-class subset of ImageNet) to make training feasible on free or consumer-grade GPU.

* **Hardware:** Trained on a single NVIDIA Tesla T4 (16GB VRAM) on Google Colab/Kaggle
* **Dataset:** ImageNette (approx. 13,000 images). The code automatically downloads and formats this
* **Resolution:** 256x256
* **Batch Size:** 12
* **Training Speed:** Approximately 1.5 iterations per second (roughly 8.5 minutes per epoch)


## usage

### 1. quick start

The easiest way to run this project is using the speedrun script. This script handles dependency installation, environment setup, and starts the training loop automatically.

```bash
bash speedrun.sh
```

### 2. manual installation

If you prefer to run commands manually, follow these steps:

First, install the package in editable mode to install dependencies (PyTorch, torchvision, wandb, einops, tqdm, numpy):

```bash
pip install -e .
```

Set your python path to include the source directory:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### 3. training

Run the training script. This will automatically download the dataset to a `./data` folder and begin training.

```bash
python src/nanojit/train.py
```

**Note on Weights & Biases (WandB):** This project uses WandB to track loss curves and visualize generated samples during training. When you run the training script for the first time, you will be prompted to log in. You should select option 2 and paste your API key from your WandB settings.

### 4. sampling / inference

To generate images using a trained checkpoint, run the sample script. You must provide the path to a valid checkpoint file (check the `results/` folder).

```bash
python src/nanojit/sample.py results/nanojit_ep50.pt
```

This will generate a grid of images (one for each class) and save it as `generated_grid.png`.

## license

MIT


