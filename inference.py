"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image

import sys
import os

from tools.utils import apply_ls_params

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engine.core import YAMLConfig

import time
import json
import argparse


class FPSLogger:
    def __init__(self, num_of_images):
        self.to_time = 0.0
        self.count = 0
        self.last_record = 0.0
        self.last_print = time.time()
        self.interval = 10
        self.num_of_images = num_of_images

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.to_time += time.time() - self.last_record
        self.count += 1
        self.print_fps()

    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - mmdet - INFO - Predict({self.count}/{self.num_of_images}) "
                  f"- Inference running at {self.count / self.to_time:.3f} FPS")
            self.last_print = time.time()


class Inference:
    def __init__(
        self,
        cfg: YAMLConfig,
        checkpoint_file: str,
        inf_dir: str,
        training_res: list = [512, 512],
        threshold: float = 0.5,
        output_file: str = None,
        device: str = 'cuda:0',
    ):
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.inf_dir = inf_dir
        self.output_file = output_file
        self.training_res = training_res
        self.cfg = cfg

        self.fps_logger = FPSLogger(len(os.listdir(self.inf_dir)))
        self.model = self.init_model()

        self.run(threshold)

    def init_model(self):
        if 'HGNetv2' in self.cfg.yaml_cfg:
            self.cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        resolution = self.training_res
        cfg_model = self.cfg.model
        cfg_preprocess = self.cfg.postprocessor
        device = self.device

        # Load train mode state and convert to deploy mode
        self.cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.model = cfg_model.deploy()
                self.size = torch.tensor([resolution]).to(device)
                self.postprocessor = cfg_preprocess.deploy()

            def forward(self, images):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, self.size)
                return outputs

        return Model().to(self.device)

    def process_image(self, file_path):
        im_pil = Image.open(file_path).convert('RGB')

        transforms = T.Compose([
            T.Resize(self.training_res),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        output = self.model(im_data)
        return output

    def run(self, threshold=0.5):
        results = []
        for imn in os.listdir(self.inf_dir):
            img_path = f"{self.inf_dir}/{imn}"
            self.fps_logger.start_record()
            labels, boxes, scores = self.process_image(img_path)
            bboxes = []
            for label, bbox, score in zip(labels[0], boxes[0], scores[0]):
                if score > threshold:
                    # x1, x2, y1, y2 why?
                    det = list(map(float, [bbox[0], bbox[2], bbox[1], bbox[3], label, score]))
                    bboxes.append(det)
            results.append({img_path: bboxes})
            self.fps_logger.end_record()
        if self.output_file:
            with open(self.output_file, "w") as f:
                json.dump(results, f)
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')

    # LabelStudio
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument(
        "--inf_dir", type=str, help="Directory containing images for inference"
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--output_file", type=str, help="Directory to save inference results"
    )

    parser.add_argument('--train-epochs', type=int, default=72)
    parser.add_argument(
        '--training-res',
        type=int,
        default=[512, 512],
        nargs='+',
        help='Training resolution')

    args = parser.parse_args()
    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg = apply_ls_params(args, cfg)

    Inference(
        cfg=cfg,
        checkpoint_file=args.resume,
        inf_dir=args.inf_dir,
        training_res=args.training_res,
        threshold=args.threshold,
        output_file=args.output_file,
        device=args.device
    )
