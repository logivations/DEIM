# if nproc_per_node != number of gpu it hang without reason
torchrun --master_port=7777 \
    --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  \
    train.py -c \
    configs/deim_dfine/dfine_hgnetv2_s_coco.yml \
    --use-amp \
    --seed=0
