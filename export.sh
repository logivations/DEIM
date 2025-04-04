pip install onnx onnxsim && \
    python tools/deployment/export_onnx.py --check \
    -c configs/deim_dfine/dfine_hgnetv2_s_coco.yml \
    -r path/to/best.pth