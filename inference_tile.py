import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple

o_path = os.getcwd()
sys.path.append(o_path)

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from einops import rearrange
from diffusers import DDIMScheduler

from models.unet import UNet3DVSRModel
from models.pipeline_stable_diffusion_DiffVSR import StableDiffusionUpscalePipeline
from models.autoencoder_kl_TE_3DVAE import AutoencoderKLTemporalDecoder
from utils import *

SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
SUPPORTED_VIDEOS = {'.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pipeline(pretrained_model: str) -> StableDiffusionUpscalePipeline:
    """
    Load and configure the inference pipeline
    """
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        './pretrained_models/upscaler4x',
        torch_dtype=torch.float16
    )

    # Load VAE model
    vae_path = "./pretrained_models/TE-3DVAE.pt"
    vae_config = "./configs/vae_config.json"
    vae_cfg = AutoencoderKLTemporalDecoder.load_config(vae_config)
    pipeline.vae = AutoencoderKLTemporalDecoder.from_config(vae_cfg)
    pipeline.vae.load_state_dict(torch.load(vae_path, map_location="cpu"), strict=True)

    # Load UNet model
    config_path = "./configs/unet_3d_config.json"
    unet_cfg = UNet3DVSRModel.load_config(config_path)
    pipeline.unet = UNet3DVSRModel.from_config(unet_cfg)
    checkpoint = torch.load(pretrained_model, map_location="cpu")['ema']
    pipeline.unet.load_state_dict(checkpoint, True)
    pipeline.unet = pipeline.unet.half()
    pipeline.unet.eval()

    # Configure scheduler
    with open('./pretrained_models/upscaler4x/scheduler/scheduler_config.json', "r") as f:
        scheduler_config = json.load(f)
    scheduler_config["beta_schedule"] = "linear"
    pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)

    return pipeline.to(DEVICE)

def process_video(vframes: torch.Tensor, pipeline: StableDiffusionUpscalePipeline, 
                 prompt: str, args: argparse.Namespace) -> torch.Tensor:
    """
    Process a single video
    """
    vframes = (vframes / 255. - 0.5) * 2  # Normalize to [-1, 1]
    vframes = vframes.float()
    orig_t, c, h, w = vframes.shape
    
    # Process frame count, ensure it's a multiple of 8
    if orig_t % 8 != 0:
        pad_frames = 8 - (orig_t % 8)
        padding = torch.stack([vframes[-(i + 2)] for i in range(pad_frames)])
        vframes = torch.cat([vframes, padding], dim=0)
        print(f'Padded frames: {pad_frames}, Current shape: {vframes.shape}')
    
    t = vframes.shape[0]
    vframes = vframes.unsqueeze(0)
    vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()

    negative_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures, worst quality"
    generator = torch.Generator(device=DEVICE).manual_seed(10)

    with torch.no_grad():
        if h * w > 400 * 400:  # Use tile processing for large videos
            # Set tile size and overlap
            tile_height = tile_width = args.tile_size
            tile_overlap = args.tile_overlap
            
            # Calculate number of tiles needed
            tiles_x = math.ceil(w / tile_width)
            tiles_y = math.ceil(h / tile_height)
            print(f'Processing the video w/ tile patches [{tiles_x}x{tiles_y}]...')
            
            # Handle boundary cases
            rm_end_pad_w, rm_end_pad_h = True, True
            if (tiles_x - 1) * tile_width + tile_overlap >= w:
                tiles_x = tiles_x - 1
                rm_end_pad_w = False
            if (tiles_y - 1) * tile_height + tile_overlap >= h:
                tiles_y = tiles_y - 1
                rm_end_pad_h = False
            
            # Initialize output tensor
            output_h, output_w = h * 4, w * 4
            output = torch.zeros((t, c, output_h, output_w), device="cpu", dtype=torch.float32)
            
            # Process each tile
            for y in range(tiles_y):
                for x in range(tiles_x):
                    print(f"\ttile: [{y+1}/{tiles_y}] x [{x+1}/{tiles_x}]")
                    
                    # Calculate current tile coordinates
                    ofs_x = x * tile_width
                    ofs_y = y * tile_height
                    
                    # Calculate input tile region
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_width, w)
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_height, h)
                    
                    # Calculate padded input tile region
                    input_start_x_pad = max(input_start_x - tile_overlap, 0)
                    input_end_x_pad = min(input_end_x + tile_overlap, w)
                    input_start_y_pad = max(input_start_y - tile_overlap, 0)
                    input_end_y_pad = min(input_end_y + tile_overlap, h)
                    
                    # Calculate tile dimensions
                    input_tile_width = input_end_x - input_start_x
                    input_tile_height = input_end_y - input_start_y
                    
                    # Extract current tile
                    input_tile = vframes[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                    
                    try:
                        # Process current tile
                        output_tile = pipeline(
                            prompt=prompt,
                            image=input_tile,
                            generator=generator,
                            num_inference_steps=args.inference_steps,
                            guidance_scale=args.guidance_scale,
                            noise_level=args.noise_level,
                            negative_prompt=negative_prompt,
                        ).images.to("cpu")
                        
                        # Calculate output tile position
                        output_start_x = input_start_x * 4
                        if x == tiles_x - 1 and rm_end_pad_w == False:
                            output_end_x = output_w
                        else:
                            output_end_x = input_end_x * 4
                        
                        output_start_y = input_start_y * 4
                        if y == tiles_y - 1 and rm_end_pad_h == False:
                            output_end_y = output_h
                        else:
                            output_end_y = input_end_y * 4
                        
                        # Calculate unpadded output tile region
                        output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                        if x == tiles_x - 1 and rm_end_pad_w == False:
                            output_end_x_tile = output_start_x_tile + output_w - output_start_x
                        else:
                            output_end_x_tile = output_start_x_tile + input_tile_width * 4
                        
                        output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                        if y == tiles_y - 1 and rm_end_pad_h == False:
                            output_end_y_tile = output_start_y_tile + output_h - output_start_y
                        else:
                            output_end_y_tile = output_start_y_tile + input_tile_height * 4
                        
                        # Place processed tile into output tensor
                        output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = (
                            output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                      output_start_x_tile:output_end_x_tile]
                        )
                    except RuntimeError as error:
                        print('Error', error)
                        continue
            
            upscaled_video = output.cpu()
            
        else:
            # Original non-tile processing logic
            upscaled_video = pipeline(
                prompt=prompt,
                image=vframes,
                generator=generator,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                noise_level=args.noise_level,
                negative_prompt=negative_prompt,
            ).images.to("cpu")

    return upscaled_video[:orig_t]

def main(args: argparse.Namespace):
    torch.set_grad_enabled(False)
    
    # Load model
    pipeline = load_pipeline(args.pretrained_model)
    
    # Get video list
    if args.input_path.endswith(tuple(SUPPORTED_VIDEOS)):
        video_list = [args.input_path]
    elif Path(args.input_path).is_dir():
        first_file = next(Path(args.input_path).iterdir())
        if first_file.suffix in SUPPORTED_IMAGES:
            video_list = [args.input_path]
        elif first_file.suffix in SUPPORTED_VIDEOS:
            video_list = get_video_paths(args.input_path)
        else:
            raise ValueError(f"Unsupported input format: {first_file.suffix}")
    else:
        raise ValueError(f"Invalid input path: {args.input_path}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Read prompt
    if args.val_prompt.lower() == 'none':
        prompt_df = None
    else:
        prompt_df = pd.read_csv(args.val_prompt, dtype={'video_path': str})

    # Process each video
    for idx, video_path in enumerate(video_list, 1):
        vframes, fps, size, video_name = read_frame_from_videos(video_path)
        print(f'[{idx}/{len(video_list)}] Processing video: {video_name}')

        save_path = Path(args.output_path) / f"{video_name}_diffvsr_n{args.noise_level}_g{args.guidance_scale}_s{args.inference_steps}.mp4"
        
        if save_path.exists():
            print(f"Skipping existing file: {save_path}")
            continue

        if prompt_df is None:
            prompt = "clear, high quality, high-resolution, 4K"
        else:
            prompt = get_prompt_by_video_name(
                video_name, 
                prompt_df, 
                name_column='video_name',
                prompt_column='caption'
            ) + " clear, high quality, high-resolution, 4K"

        print(f"Video name: {video_name}\nPrompt: {prompt}")

        start_time = time.time()
        upscaled_video = process_video(vframes, pipeline, prompt, args)
        process_time = time.time() - start_time

        # Post-processing and saving
        h, w = vframes.shape[-2:]
        upscaled_video = F.interpolate(
            upscaled_video,
            size=(h * 4, w * 4),
            mode='bilinear',
            align_corners=False
        )
        upscaled_video = (upscaled_video / 2 + 0.5).clamp(0, 1) * 255

        if args.outputimage_path:
            outimage_path = Path(args.outputimage_path) / video_name
            split_and_save_images(upscaled_video, outimage_path)

        # Save video
        upscaled_video = upscaled_video.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        save_video(upscaled_video, str(save_path), fps=8, use_ffmpeg=args.use_ffmpeg)

        print(f'Saved processed video "{video_name}" to {save_path}, Processing time: {process_time:.2f}s\n')

    print(f'\nAll results saved to {args.output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Super Resolution Processing Tool')
    parser.add_argument('-i', '--input_path', type=str, default='./input',
                      help='Input path (can be video file, image folder, or video folder)')
    parser.add_argument('-o', '--output_path', type=str, default='./output',
                      help='Output path')
    parser.add_argument('-n', '--noise_level', type=int, default=50,
                      help='Noise level')
    parser.add_argument('-g', '--guidance_scale', type=int, default=5,
                      help='Guidance scale')
    parser.add_argument('-s', '--inference_steps', type=int, default=50,
                      help='Number of inference steps')
    parser.add_argument('-p', '--pretrained_model', type=str, default='./pretrained_models/DiffVSR_UNet.pt',
                      help='Path to pretrained model')
    parser.add_argument('-txt', '--val_prompt', type=str, default='none',
                      help='Path to prompt file')
    parser.add_argument('-oimg', '--outputimage_path', type=str, default=None,
                      help='Path to save individual frames')
    parser.add_argument('--use_ffmpeg', action='store_true', default=False,
                      help='Use ffmpeg to encode output video')
    parser.add_argument('--tile_size', type=int, default=256,
                      help='Tile size (height/width) used during tiled processing (default 256)')
    parser.add_argument('--tile_overlap', type=int, default=64,
                      help='Overlap in pixels between tiles during tiled processing (default 64)')
    
    args = parser.parse_args()
    main(args)
