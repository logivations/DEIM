"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig
from tools.utils import apply_ls_params


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg = apply_ls_params(args, cfg)
    output_file = None
    if args.resume:
        if args.resume.endswith('best.pth'):
            output_file = args.resume.replace('best.pth', 'dfine_model.onnx')
            if os.path.exists(args.resume.replace('best.pth', 'best_stg2.pth')):
                args.resume = args.resume.replace('best.pth', 'best_stg2.pth')
            elif os.path.exists(args.resume.replace('best.pth', 'best_stg1.pth')):
                args.resume = args.resume.replace('best.pth', 'best_stg1.pth')
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.size = torch.tensor([args.training_res])

            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, self.size)
            return outputs

    model = Model()

    data = torch.rand(32, 3, *args.training_res)
    size = torch.tensor([args.training_res])
    _ = model(data)

    dynamic_axes = {
        'images': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
        'boxes': {
            0: 'batch',
            1: 'num_dets',
        },
        'scores': {
            0: 'batch',
            1: 'num_dets',
        },
    }
    if output_file is None:
        output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    torch.onnx.export(
        model,
        (data),
        output_file,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--check',  action='store_true', default=True,)
    parser.add_argument('--simplify',  action='store_true', default=True,)
    parser.add_argument('--train-epochs', type=int, default=72)
    parser.add_argument(
        '--training-res',
        type=int,
        default=[512, 512],
        nargs='+',
        help='Training resolution')

    args = parser.parse_args()
    main(args)
