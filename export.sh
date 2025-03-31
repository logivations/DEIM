pip install onnx onnxsim && \
    python tools/deployment/export_onnx.py --check \
    -c configs/deim_dfine/dfine_hgnetv2_s_coco.yml \
    -r output/dfine_orig_config_fix_classes/best_stg2.pth