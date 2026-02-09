torchrun --master_port=7777 --nproc_per_node=1 train.py\
 -c configs/label_studio/ls_dfine_hgnetv2_s_coco.yml\
 --train-epochs 72\
 --training-res 512 512\
 --use-amp --seed=0 &&\
 python3 tools/deployment/export_onnx.py\
 --check -c configs/label_studio/ls_dfine_hgnetv2_s_coco.yml\
 --training-res 512 512 -r /result/best.pth &&\
 bash make_nvinfer_config.sh\
 --nvinfer-file /result/object-config.txt\
 --onnx-filename dfine_model.onnx\
 --classes amr electric_pallet_jack forklift low_forklift manual_pallet_jack other_vehicle person roll_cage tugger\
 --res 512 512