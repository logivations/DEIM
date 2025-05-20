#!/bin/bash

NVINFER_FILE=""
ONNX_FILENAME=""
CLASSES=()
RES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvinfer-file)
            NVINFER_FILE="$2"
            shift 2
            ;;
        --onnx-filename)
            ONNX_FILE_NAME="$2"
            shift 2
            ;;
        --classes)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --res)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                RES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")
RES=$(IFS=';' ; echo "${RES[*]}")

echo "[property]
# preprocessing parameters.
net-scale-factor=0.00392156862
offsets=0;0;0
model-color-format=1
# 0=Nearest, 1=Bilinear 2=VIC-5 Tap interpolation 3=VIC-10 Tap interpolation
scaling-filter=3

onnx-file=$ONNX_FILE_NAME
infer-dims=3;$RES

[custom]
# 1 - PVT, 2 - DEIM, 3 - TAO
detector-type=2
min_confidence = 0.5
labels=$CLASSES
report-labels=$CLASSES
" > "$NVINFER_FILE"
