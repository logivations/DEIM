docker run --network host -w /DEIM \
  -v /data/DEIM:/DEIM \
  -v /data/dd_dataset:/data/dd_dataset \
  --gpus '"device=0"' --shm-size=8g -it quay.io/logivations/ml_all:deim
