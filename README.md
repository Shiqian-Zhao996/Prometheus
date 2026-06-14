# Prometheus

This repository contains the official implementation of **"Towards Effective Prompt Stealing Attack Against Text-to-Image Diffusion Models"**, accepted by **NDSS 2026**.

If you find this repository useful, please consider starring it.

## 1. Download The Code

Clone the repository from GitHub:

```bash
git clone https://github.com/Shiqian-Zhao996/Prometheus.git
cd Prometheus
```

## 2. Install The Conda Environment

Create the conda environment from the provided environment file. By default, the environment name is `prometheus`, but you can customize it by editing the `name:` field in `prometheus_environment.yml` before running the command.

```bash
conda env create -f prometheus_environment.yml
conda activate prometheus
```

If you already created the environment and want to update it:

```bash
conda env update -n prometheus -f prometheus_environment.yml --prune
conda activate prometheus
```

## 3. Download Required Models

### BLIP Caption Models

Prometheus uses BLIP for image captioning. The BLIP implementation and checkpoints come from the official [Salesforce BLIP repository](https://github.com/salesforce/BLIP).

BLIP provides two official caption checkpoints:

1. Base caption model:
   https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth

2. Large caption model:
   https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth

The current release uses the base caption model by default. Download it into the existing `data/` directory:

```bash
wget -P data https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
```

The current code loads:

```text
data/model_base_caption_capfilt_large.pth
```

The large checkpoint is an official alternative for users who want to adapt the BLIP caption model configuration for additional experiments.

### spaCy NLP Model

Prometheus uses the spaCy transformer English model for parsing generated captions. Install it inside the `prometheus` environment:

```bash
python -m spacy download en_core_web_trf
```

This model is intentionally installed after environment creation instead of being listed in `prometheus_environment.yml`, because spaCy model packages are distributed separately and may fail during `conda env create` on some Python versions.

### Oracle Diffusion Models

The oracle models are loaded through Hugging Face Diffusers:

- `stabilityai/stable-diffusion-3.5-large-turbo`
- `black-forest-labs/FLUX.1-schnell`
- `shuttleai/shuttle-3-diffusion`

Diffusers will download these models on first use. Make sure your environment has network access and, if required by the model provider, that you have accepted the model license and logged in to Hugging Face. See the official Hugging Face CLI login documentation: https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login

```bash
hf auth login
```

## 4. Test The Running Code

Run the provided script:

```bash
sh script/main.sh
```

Or run `main.py` directly:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --oracle ShuttleDiffusion \
  --dataset DALLEPrompt \
  --max_budget 200
```

To accelerate the evaluation, you can reduce the query budget to `100`:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --oracle ShuttleDiffusion \
  --dataset DALLEPrompt \
  --max_budget 100
```

This significantly reduces runtime while usually causing only a limited performance reduction.

For this dataset, use:

```bash
--dataset DALLEPrompt
```

Available arguments:

- `--oracle`: one of `SD3.5`, `FLUX`, `ShuttleDiffusion`
- `--dataset`: currently `DALLEPrompt`
- `--max_budget`: query budget for prompt stealing
- `--caption_repeat`: number of sampled captions for dynamic modifier discovery
- `--resolution`: input image resolution

The current implementation prints per-prompt results and final averages for `CLIP-II`, `LPIPS`, `SBERT`, and `ASR` at the end of execution. It does not write result files to disk.

## Citation

If you feel this repo is useful, please cite:

```bibtex
@article{zhao2025towards,
  title={Towards effective prompt stealing attack against text-to-image diffusion models},
  author={Zhao, Shiqian and Wang, Chong and Li, Yiming and Huang, Yihao and Qu, Wenjie and Lam, Siew-Kei and Xie, Yi and Chen, Kangjie and Zhang, Jie and Zhang, Tianwei},
  journal={arXiv preprint arXiv:2508.06837},
  year={2025}
}
```
