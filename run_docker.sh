docker run --name deim_train \
  --network host -w /DEIM \
  -v /data/DEIM:/DEIM \
  -v /data:/data \
  --gpus '"device=0"' --shm-size=8g -it quay.io/logivations/ml_all:deim
