docker run --name deim_train \
  --network host -w /DEIM \
  -v /data/DEIM:/DEIM \
  -v /data/DEIM/.cache/torch:/root/.cache/torch \
  -v /data/DEIM/.cache/triton:/root/.triton/cache \
  -v /data/DEIM/.cache/nv:/root/.nv \
  --rm --gpus all \
  --shm-size=8g -it quay.io/logivations/ml_all:LS_dfine_latest
