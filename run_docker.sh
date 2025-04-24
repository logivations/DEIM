docker run --name deim_train \
  --network host -w /DEIM \
  -v /data/DEIM:/DEIM \
  -v /mnt/nfs_training:/datasets \
  --rm --gpus all \
  --shm-size=8g -it quay.io/logivations/ml_all:LS_dfine_latest
