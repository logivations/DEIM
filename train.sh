CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=4 train.py -c \
configs/deim_dfine/dfine_hgnetv2_s_coco.yml --use-amp --seed=0
