import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import ffmpeg
import imageio

# 定义常量
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        video_name = os.path.basename(frame_root)[:-4]
        frames, _, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', output_format='TCHW') # RGB
        fps = info['video_fps']
    else:
        clean_root = os.path.normpath(frame_root)
        video_name = os.path.basename(clean_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[...,[2,1,0]] # RGB, HWC
            frames.append(frame)
        fps = None
        frames = torch.Tensor(np.array(frames)).permute(0, 3, 1, 2).contiguous() # TCHW
    size = frames[0].size

    return frames, fps, size, video_name

def get_video_paths(input_root):
    video_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)

def split_and_save_images(upscaled_videos, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    T, C, H, W = upscaled_videos.shape

    for t in range(T):
        single_frame = upscaled_videos[t, :, :, :]
        image_data = single_frame.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image_data.transpose(1, 2, 0))
        image.save(os.path.join(output_folder, f'frame_{t:03d}.png'))


def get_prompt_by_video_name(video_name, df, name_column='video_name', prompt_column='caption'):
    """
    从DataFrame中获取视频对应的prompt
    
    Args:
        video_name: 视频名称
        df: DataFrame包含视频信息
        name_column: 视频名称列的列名
        prompt_column: prompt列的列名
    
    Returns:
        str or None: 找到的prompt或None
        
        prompt_column  sampled_frame_caption
    """
    row = df[df[name_column] == str(video_name)]
    return row.iloc[0][prompt_column] if not row.empty else None

def save_video(video_frames: np.ndarray, save_path: str, fps: int = 8, use_ffmpeg: bool = False) -> None:
    """
    Save frames to a video file.
    Args:
        video_frames: numpy array with shape (T, H, W, C)
        save_path: output video path
        fps: frames per second
        use_ffmpeg: whether to use ffmpeg to encode
    """
    if use_ffmpeg:
        try:
            temp_folder = os.path.join(os.path.dirname(save_path), "temp_frames")
            os.makedirs(temp_folder, exist_ok=True)
            for i, frame in enumerate(video_frames):
                frame_path = os.path.join(temp_folder, f"frame_{i:03d}.png")
                Image.fromarray(frame).save(frame_path)
            input_pattern = os.path.join(temp_folder, "frame_%03d.png")
            stream = (
                ffmpeg
                .input(input_pattern, pattern_type='sequence', start_number=0, framerate=fps)
                .output(save_path, codec='libx264', preset='slower', crf=23, video_bitrate='2M', pix_fmt='yuv420p', loglevel='error')
                .overwrite_output()
            )
            stream.run(capture_stdout=True, capture_stderr=True)
            import shutil
            shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"Error using ffmpeg to save video: {str(e)}. Fallback to imageio.mimwrite.")
            imageio.mimwrite(save_path, video_frames, fps=fps, quality=9)
    else:
        imageio.mimwrite(save_path, video_frames, fps=fps, quality=9)
