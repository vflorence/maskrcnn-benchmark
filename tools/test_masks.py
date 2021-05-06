from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug

import logging
import tempfile
from collections import OrderedDict
from tqdm import tqdm
import time

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')



def evaluate(dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset)
    
    output = {}
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            output[iou_type] = evaluate_predictions_on_coco(dataset.coco, coco_results[iou_type],file_path, iou_type)
    return output


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'keypoints': keypoint,
            'score': scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)



def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        coco_results = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        

        from collections import defaultdict
        import pycocotools.mask as maskutils
        from maskrcnn_benchmark.utils import cv2_util
        import cv2
        import skimage.io as io
        import numpy as np
        import matplotlib.pyplot as plt 
        import IPython
        
        LVIS_CATEGORIES = [
            {'name': 'background', 'id': 0, 'lvis': 0},   
            {'name': 'apple', 'id': 1, 'lvis':13},
            {'name': 'banana', 'id': 2, 'lvis':45},
            {'name': 'pitcher', 'id': 3, 'lvis':827},
            {'name': 'wine glass', 'id': 4, 'lvis':1215},
            {'name': 'bowl', 'id': 5, 'lvis':139},
            {'name': 'mug', 'id': 6, 'lvis':720},
            {'name': 'sponge', 'id': 7, 'lvis':1017},
            {'name': 'plate', 'id': 8, 'lvis':831},
            {'name': 'fork', 'id': 9, 'lvis':475},
            {'name': 'knife', 'id': 10, 'lvis':625},
            {'name': 'spoon', 'id': 11, 'lvis':1018},
            {'name': 'spatula', 'id': 12, 'lvis':1012},
            {'name': 'table cloth', 'id': 13, 'lvis':1072},
            {'name': 'power drill', 'id': 14, 'lvis':401},
            {'name': 'scissors', 'id': 15, 'lvis':931},
            {'name': 'padlock', 'id': 16, 'lvis':758},
            {'name': 'expo', 'id': 17, 'lvis':682},
            {'name': 'wrench', 'id': 18, 'lvis':1222},
            {'name': 'screwdriver', 'id': 19, 'lvis':942},
            {'name': 'soccer ball', 'id': 20, 'lvis':997},
            {'name': 'baseball', 'id': 21, 'lvis':57},
            {'name': 'magazine', 'id': 22, 'lvis':672},
            {'name': 'tshirt', 'id': 23, 'lvis':965},
            {'name': 'keys', 'id': 24, 'lvis':615},
            {'name': 'hammer', 'id': 25, 'lvis':540}
        ]
        LVIS_CATEGORIES = {c['id']: c['name'] for c in LVIS_CATEGORIES}
        
        
        vis_dir = os.path.join(output_folder,'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        template = "{}: {:.2f}"
        
        segm = coco_results['segm'].evalImgs
        bbox = coco_results['bbox'].evalImgs
        
        segm = [r for r in segm if((not r is None) and (r['aRng'][0] == 0. and r['aRng'][1] == 10000000000.))] 
        bbox = [r for r in bbox if((not r is None) and (r['aRng'][0] == 0. and r['aRng'][1] == 10000000000.))] 
        
        image_segm = defaultdict(list)
        image_bbox = defaultdict(list)
        
        for r in segm:
            image_segm[r['image_id']].append(r)
        
        for i in range(0, max(len(image_segm), len(image_bbox))):
            # Per Image
            path = data_loader_val.dataset.get_img_info(i)['file_name']
            result = io.imread(path)
            scores = []; matched = []; annos = []; ignore = []; 
            for s in image_segm[i]:
                scores.extend(s['dtScores'])
                matched.extend(s['dtMatches'][0,:])
                ignore.extend(s['dtIgnore'][0,:])
                annos.extend(s['dtIds'])
                
            anns = sorted(zip(matched,scores,annos,ignore), key=lambda t:t[0])
            
            for a in anns:
                #Per annotation - unmatched to matched
                match, score, anno_id, ignore = a
                if ignore:
                    continue
                
                segm_anno = coco_results['segm'].cocoDt.anns[anno_id]
                bbox_anno = coco_results['bbox'].cocoDt.anns[anno_id]
                category_id = segm_anno['category_id']
                
                category_label = LVIS_CATEGORIES[category_id] 
                
                if match > 0:
                    gt_id = coco_results['segm'].cocoGt.anns[match]['category_id']
                    print('dt_cat: ',category_id, '    gt_cat: ', gt_id)
                    color = (category_id*palette%255).numpy().astype("uint8").tolist()
                    thickness = 2
                else:
                    color = (255,0,0)
                    thickness=1
                    
                
                mask = maskutils.decode(segm_anno['segmentation'])
                contours, hierarchy = cv2_util.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                result = cv2.drawContours(result,contours, -1, color, thickness)
                
                box = bbox_anno['bbox']
                box = [round(b) for b in box]
                result = cv2.rectangle(
                    result, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), tuple(color))
                
                x, y = box[:2]
                s = template.format(category_label, score)
                if match > 0:
                    cv2.putText(
                        result, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1
                    )
                else:
                    cv2.putText(
                        result, s, (x, y+box[3]+12), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1
                    )
            io.imsave(os.path.join(vis_dir,'result%d.png'%i), result)
                
        IPython.embed()

        #results = defaultdict(list)
        #for s in coco_results['segm']:
            #results[s['image_id']].append(s)
        
        #for i,r_im in results.items():
            #image_path = data_loader_val.dataset.get_img_info(i)['file_name']
            #image = io.imread(image_path)
            
            #for j,result in enumerate(r_im):
                #mask = maskutils.decode(result['segmentation'])
                #contours, hierarchy = cv2_util.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #image = cv2.drawContours(image,contours, -1, (result['category_id']*palette%255).numpy().astype("uint8").tolist(), 3)
            #import IPython; IPython.embed()    
            
	# overlay bbox, segm, class (see demo/predictor.py for an example)


if __name__ == "__main__":
    main()
