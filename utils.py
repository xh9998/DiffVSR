import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

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
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[...,[2,1,0]] # RGB, HWC
            frames.append(frame)
        fps = None
        frames = torch.Tensor(np.array(frames)).permute(0, 3, 1, 2).contiguous() # TCHW
    size = frames[0].size

    frames = frames.to(device)
    
    return frames, fps, size, video_name


def split_and_save_images(upscaled_videos, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    T, C, H, W = upscaled_videos.shape

    for t in range(T):
        single_frame = upscaled_videos[t, :, :, :]
        image_data = single_frame.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image_data.transpose(1, 2, 0))
        image.save(os.path.join(output_folder, f'frame_{t:03d}.png'))


def get_prompt_by_video_name(video_name, df, name_column='video_name', prompt_column='sampled_frame_caption'):
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