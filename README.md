<div align="center">

<h1>
  DiffVSR:<br>
  Temporal-Consistent Diffusion for Real-World Video Super-Resolution
</h1>

<h4 align="center">
  <!-- Fill these links with your own resources -->
  <a href="https://xh9998.github.io/DiffVSR-project/" target="_blank">
    <img src="https://img.shields.io/badge/🐳-Project%20Page-blue" />
  </a>
  <a href="https://arxiv.org/abs/2501.10110" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-DiffVSR-b31b1b.svg" />
  </a>
  <a href="https://www.youtube.com/embed/ezRM2xF3fDw" target="_blank">
    <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white" />
  </a>
  <a href="https://huggingface.co/huihui9998/DiffVSR/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/🤗-HuggingFace%20Models-yellow" />
  </a>
</h4>

<strong>DiffVSR is a diffusion-based upscaler for videos, conditioned on LR video and text prompts, with temporal consistency.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
  <img style="width:100%" src="./assets/DiffVSR_method.png" alt="Method Overview" />
</div>

:open_book: For more visual results, visit the project page.

</div>

---

## 🔥 Update
- [2025.09] Initial public release of inference toolkit and README.
- [2025.09] Added ffmpeg-based video writer (`--use_ffmpeg`).


<!-- ## 🧩 ILT Illustration
<p align="center">
  <img src="./assets/ILT.png" alt="ILT Illustration" />
</p> -->


## 🔧 Dependencies and Installation
We provide a conda environment file. Create and activate it:
```bash
conda env create -f DiffVSR_env.yml
conda activate DiffVSR
```
Key packages (see `DiffVSR_env.yml` for the full list): PyTorch 2.0.0 (CUDA 11.7), diffusers 0.30.0, torchvision 0.15.0, einops, opencv, pandas, rotary-embedding-torch, xformers (optional), imageio.

## 📂 Pretrained Models & Configs
Place/check the following files (paths are used by the code):
- VAE (temporal): `./pretrained_models/TE-3DVAE.pt`
- UNet3D checkpoint: passed via `-p/--pretrained_model`
-

Directory example:
```
DiffVSR/
 ├─ inference_tile.py
 ├─ DiffVSR_env.yml
 ├─ configs/
 │   ├─ unet_3d_config.json
 │   └─ vae_config.json
 └─ pretrained_models/
     ├─ TE-3DVAE.pt
     └─ upscaler4x/
         └─ scheduler/
             └─ scheduler_config.json
```

## ☕️ Quick Inference
`--input_path` can be a single video, a frames folder, or a folder of videos.

```bash
python inference_tile.py \
  -i ./test_video/Aurora \
  -o ./output \
  -txt /path/to/captions.csv \
  -p /path/to/unet_checkpoint.pt \
  -n 50 -g 5 -s 50 \
  --use_ffmpeg
```

Arguments (main):
- `-i/--input_path`: video file, frames folder (avoid trailing slash if name becomes empty), or folder of videos
- `-o/--output_path`: directory for output mp4
- `-txt/--val_prompt`: CSV containing `video_path` and `sampled_frame_caption`
- `-p/--pretrained_model`: UNet checkpoint path
- `-n/--noise_level`: noise level (default 50)
- `-g/--guidance_scale`: guidance scale (default 5)
- `-s/--inference_steps`: denoising steps (default 50)
- `-oimg/--outputimage_path`: dump generated PNG frames when provided
- `--use_ffmpeg`: use ffmpeg for video encoding (fallback to imageio on error)

## 🧩 CSV Prompt Format
`--val_prompt` CSV should include one row per video with columns:
- `video_path`: base name matching input (e.g., folder name or file stem)
- `sampled_frame_caption`: positive prompt text

The script concatenates an internal positive prompt string.


## 📑 Citation
If you find this repo useful, please consider citing (fill your bibtex):
```bibtex
@article{your2025diffvsr,
  title={{DiffVSR}: Temporal-Consistent Diffusion for Real-World Video Super-Resolution},
  author={Your Name and Coauthors},
  journal={arXiv preprint arXiv:XXYY.ZZZZZ},
  year={2025}
}
```


