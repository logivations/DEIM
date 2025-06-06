"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


class CustomCOCOEvaluator(COCOeval_faster):

    def __init__(self, *args, **kwargs):
        super(CustomCOCOEvaluator, self).__init__(*args, **kwargs)

    def summarize(self):
        self.print_function(f"### Summary ###")
        num_predictions = (
            len(self.cocoDt.anns) if self.cocoDt else 0
        )
        self.print_function(f"Number of Predictions Used: {num_predictions}")
        self.print_function(f"Number of Ground Truth Annotations: {len(self.cocoGt.anns)}")
        self.print_function(f"Number of Evaluated Images: {len(self.params.imgIds)}")
        super(CustomCOCOEvaluator, self).summarize()

    def debug_print_gt_and_predictions(self):
        self.print_function("===== Ground Truth Annotations =====")
        for ann_id, ann in self.cocoGt.anns.items():
            self.print_function(f"Image ID: {ann['image_id']}, Category ID: {ann['category_id']}, BBox: {ann['bbox']}")

        self.print_function("\n===== Predictions per Image =====")
        for ann_id, ann in self.cocoDt.anns.items():
             self.print_function(f"Image ID: {ann['image_id']}, Category ID: {ann['category_id']}, BBox: {ann.get('bbox', 'N/A')}, Score: {ann.get('score', 'N/A')}")

    def calculate_metrics(self, iou_thrs: list = [0.7]):
        iou_thrs = np.array(iou_thrs)
        annotation_counts = self._get_annotation_counts()
        number_of_all_annotations = sum(annotation_counts.values())

        overall_f1 = []
        overall_map = []

        for iou_thr in iou_thrs:
            iou_idx = self._get_iou_index(iou_thr)
            precisions = self._get_precisions(iou_idx)
            recalls = self._get_recalls(iou_idx)
            f1_scores = self._compute_f1_scores(precisions, recalls)

            cat_id_to_name = self._get_cat_id_to_name()

            classes_f1, classes_map = self._print_and_compute_class_metrics(
                iou_thr, precisions, recalls, f1_scores,
                annotation_counts, number_of_all_annotations,
                cat_id_to_name
            )

            overall_f1.append(np.sum(classes_f1))
            overall_map.append(np.sum(classes_map))

        overall_f1 = np.array(overall_f1)
        overall_map = np.array(overall_map)

        self.print_function(
            f"Final mAP: {np.mean(overall_map):.4f}, Final F1 Score: {np.mean(overall_f1):.4f}"
        )
        return np.mean(overall_map), np.mean(overall_f1)

    def _get_annotation_counts(self):
        return {
            cat_id: len(self.cocoGt.getAnnIds(catIds=[cat_id]))
            for cat_id in self.params.catIds
        }

    def _get_iou_index(self, iou_thr):
        return np.where(np.isclose(self.params.iouThrs, iou_thr))[0][0]

    def _get_precisions(self, iou_idx):
        precisions = self.eval["precision"][iou_idx, :, :, 0, -1]  # shape (R, K)
        # mean skipping -1
        mean_precisions_per_class = np.mean(
            np.where(precisions == -1, np.nan, precisions), axis=0
        )
        return mean_precisions_per_class

    def _get_recalls(self, iou_idx):
        recalls = self.eval["recall"][iou_idx, :, 0, -1]  # shape (K,)
        recalls = np.where(recalls == -1, np.nan, recalls)
        return recalls

    def _compute_f1_scores(self, precisions, recalls):
        return 2 * precisions * recalls / (precisions + recalls)

    def _get_cat_id_to_name(self):
        return {cat["id"]: cat["name"] for cat in self.cocoGt.cats.values()}

    def _print_and_compute_class_metrics(
        self, iou_thr, precisions, recalls, f1_scores,
        annotation_counts, number_of_all_annotations,
        cat_id_to_name
    ):
        self.print_function(f"### F1-Score for IoU threshold {iou_thr} ###")
        self.print_function(
            f"{'Label':<25} {'Num of Ann':<12} {'AR':<10} {'AP':<10} {'f1-score':<10}"
        )

        classes_f1 = []
        classes_map = []

        for idx, cat_id in enumerate(self.params.catIds):
            name = cat_id_to_name[cat_id]
            precision = precisions[idx]
            recall = recalls[idx]
            f1 = f1_scores[idx]

            num_annotations = annotation_counts.get(cat_id, 1)
            adjusted_f1 = 0.0
            adjusted_map = 0.0

            if num_annotations != 0:
                adjusted_f1 = f1 * (num_annotations / number_of_all_annotations)
                adjusted_map = precision * (num_annotations / number_of_all_annotations)

            self.print_function(
                f"{name:<25} "
                f"{num_annotations:<12} "
                f"{str(round(recall, 4)):<10} "
                f"{str(round(precision, 4)):<10} "
                f"{str(round(f1, 4)):<10}"
            )

            if np.isnan(adjusted_f1):
                adjusted_f1 = 0.0
            if np.isnan(adjusted_map):
                adjusted_map = 0.0

            classes_f1.append(adjusted_f1)
            classes_map.append(adjusted_map)

        return classes_f1, classes_map

@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = CustomCOCOEvaluator(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = CustomCOCOEvaluator(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            coco_eval.calculate_metrics()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

    def get_image_id(self, image_name):
        """"""
        return next(
            (i_id for i_id, i_info in self.coco_gt.imgs.items() if i_info['file_name'] == image_name),
            None
        )

    def evaluated(self):
        """"""
        is_evaluated = [hasattr(coco_eval, "_evalImgs_cpp") for coco_eval in self.coco_eval.values()]
        return all(is_evaluated)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
