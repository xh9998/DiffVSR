CUDA_VISIBLE_DEVICES=0 python inference_tile.py \
-i /mnt/shared-storage-user/lixiaohui/code/test_video/Aurora/ \
-o ./output/ \
-txt /mnt/nas/vsr_test/CSV/RealVideo10_caption.csv \
--use_ffmpeg