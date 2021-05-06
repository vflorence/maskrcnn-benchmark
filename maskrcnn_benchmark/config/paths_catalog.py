# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "hsrc_srn_COCO_fgbg_train": {
        "img_dir": "hsrc_srn/JPEGImages/COCO_fgbg_train", "ann_file": "hsrc_srn/Annotations/cl_instances_COCO_fgbg_train.json"
        },

        "hsrc_srn_lab_bg_val": {
        "img_dir": "hsrc_srn/JPEGImages/lab_bg_val", "ann_file": "hsrc_srn/Annotations/cl_instances_lab_bg_val.json"
        },

        "hsrc_srn_COCO_bg_val": {
        "img_dir": "hsrc_srn/JPEGImages/COCO_bg_val", "ann_file": "hsrc_srn/Annotations/cl_instances_COCO_bg_val.json"
        },

        "hsrc_srn_orig_val": {
        "img_dir": "hsrc_srn/JPEGImages/orig_val", "ann_file": "hsrc_srn/Annotations/cl_instances_orig_val.json"
        },

        "hsrc_srn_lab_bg_train": {
        "img_dir": "hsrc_srn/JPEGImages/lab_bg_train", "ann_file": "hsrc_srn/Annotations/cl_instances_lab_bg_train.json"
        },

        "hsrc_srn_lab_fgbg_val": {
        "img_dir": "hsrc_srn/JPEGImages/lab_fgbg_val", "ann_file": "hsrc_srn/Annotations/cl_instances_lab_fgbg_val.json"
        },

        "hsrc_srn_lab_fgbg_train": {
        "img_dir": "hsrc_srn/JPEGImages/lab_fgbg_train", "ann_file": "hsrc_srn/Annotations/cl_instances_lab_fgbg_train.json"
        },

        "hsrc_srn_COCO_bg_train": {
        "img_dir": "hsrc_srn/JPEGImages/COCO_bg_train", "ann_file": "hsrc_srn/Annotations/cl_instances_COCO_bg_train.json"
        },

        "hsrc_srn_orig_train": {
        "img_dir": "hsrc_srn/JPEGImages/orig_train", "ann_file": "hsrc_srn/Annotations/cl_instances_orig_train.json"
        },

        "hsrc_srn_fg_val": {
        "img_dir": "hsrc_srn/JPEGImages/fg_val", "ann_file": "hsrc_srn/Annotations/cl_instances_fg_val.json"
        },

        "hsrc_srn_COCO_fgbg_val": {
        "img_dir": "hsrc_srn/JPEGImages/COCO_fgbg_val", "ann_file": "hsrc_srn/Annotations/cl_instances_COCO_fgbg_val.json"
        },

        "hsrc_srn_fg_train": {
        "img_dir": "hsrc_srn/JPEGImages/fg_train", "ann_file": "hsrc_srn/Annotations/cl_instances_fg_train.json"
        },
        "srn_objects_orig_train": {
            "img_dir": "YCB/JPEGImages/all_orig_train",
            "ann_file": "YCB/Annotations/all_orig_train_instances.json"
        },
        "srn_objects_orig_val": {
            "img_dir": "YCB/JPEGImages/all_orig_val",
            "ann_file": "YCB/Annotations/all_orig_val_instances.json"
        },
        "srn_orig_train": {
            "img_dir": "SRN/JPEGImages/orig-train",
            "ann_file": "SRN/Annotations/orig_gripper_train_instances.json"
        },
        "srn_orig_val": {
            "img_dir": "SRN/JPEGImages/orig-val",
            "ann_file": "SRN/Annotations/orig_gripper_val_instances.json"
        },
        "lvis_train": {
            "img_dir": "LVIS/JPEGImages/train",
            "ann_file": "LVIS/Annotations/lvis_v0.5_cocostyle_train.json"
        },
        "lvis_val": {
            "img_dir": "LVIS/JPEGImages/val",
            "ann_file": "LVIS/Annotations/lvis_v0.5_cocostyle_val.json"
        },
        "hsr_IROS_obj": {
            "img_dir": "YCB/JPEGImages/IROS",
            "ann_file": "YCB/Annotations/IROS_instances.json"
        },
        "hsr_ICRA_obj": {
            "img_dir": "YCB/JPEGImages/ICRA",
            "ann_file": "YCB/Annotations/ICRA_instances.json"
        },
        "hsr_IROS_obj_multi": {
            "img_dir": "YCB/JPEGImages/IROS_multi",
            "ann_file": "YCB/Annotations/IROS_multi_instances.json"
        },
        "hsr_ICRA_obj_multi": {
            "img_dir": "YCB/JPEGImages/ICRA_multi",
            "ann_file": "YCB/Annotations/ICRA_multi_instances.json"
        },
        "hsr_IROS_obj_orig": {
            "img_dir": "YCB/JPEGImages/IROS_orig",
            "ann_file": "YCB/Annotations/IROS_objects_orig_instances.json"
        },
        "hsr_ICRA_obj_orig": {
            "img_dir": "YCB/JPEGImages/ICRA_orig",
            "ann_file": "YCB/Annotations/ICRA_objects_orig_instances.json"
        },
        "hsr_IROS_test": {
            "img_dir": "YCB/JPEGImages/IROS_test",
            "ann_file": "YCB/Annotations/IROS_test_instances.json"
        },
        "hsr_ICRA_test": {
            "img_dir": "YCB/JPEGImages/ICRA_test",
            "ann_file": "YCB/Annotations/ICRA_test_instances.json"
        },
        "hsr_challenge5": {
            "img_dir": "challenge5/JPEGImages",
            "ann_file": "challenge5/instances_challenge5.json"
        },
        "hsr_challenge_objects": {
            "img_dir": "challenge_objects/JPEGImages",
            "ann_file": "challenge_objects/instances_challenge_objects.json"
        },
        "hsr_obj_gt_train": {
            "img_dir": "hsr/in_hand/JPEGImages",
            "ann_file": "hsr/in_hand/Annotations/instances_in_hand_object_ground_truth.json"
        },
        "hsr_obj_self_supervised_train": {
            "img_dir": "hsr/in_hand/JPEGImages",
            "ann_file": "hsr/in_hand/Annotations/instances_in_hand_object_self_supervised.json"
        },
        "hsr_obj_test": {
            "img_dir": "hsr/test/JPEGImages",
            "ann_file": "hsr/test/Annotations/instances_test_object.json"
        },
        "hsr_gripper_train_ICRA":{
            "img_dir": "hsr/gripper_ICRA/train/JPEGImages",
            "ann_file": "hsr/gripper_ICRA/train/instances_gripper.json"
        },
        "hsr_gripper_val_ICRA":{
            "img_dir": "hsr/gripper_ICRA/val/JPEGImages",
            "ann_file": "hsr/gripper_ICRA/val/instances_gripper.json"
        },
        "hsr_gripper_train": {
            "img_dir": "hsr/gripper/train/JPEGImages",
            "ann_file": "hsr/gripper/train/instances_gripper.json"
        },
        "hsr_gripper_val": {
            "img_dir": "hsr/gripper/val/JPEGImages",
            "ann_file": "hsr/gripper/val/instances_gripper.json"
        },
        "hsr_gripper_test": {
            "img_dir": "hsr/gripper/test/JPEGImages",
            "ann_file": "hsr/gripper/test/instances_gripper.json"
        },
        "hsr_gripper_bg_train":{
            "img_dir": "hsr/gripper_bg/train/JPEGImages",
            "ann_file": "hsr/gripper_bg/train/instances_gripper_bg.json"
        },
        "hsr_gripper_bg_val":{
            "img_dir": "hsr/gripper_bg/val/JPEGImages",
            "ann_file": "hsr/gripper_bg/val/instances_gripper_bg.json"
        },
        "hsr_gripper_bg_test":{
            "img_dir": "hsr/gripper_bg/test/JPEGImages",
            "ann_file": "hsr/gripper_bg/test/instances_gripper_bg.json"
        },
        "hsr_gripper_fg_bg_train":{
            "img_dir": "hsr/gripper_fg_bg/train/JPEGImages",
            "ann_file": "hsr/gripper_fg_bg/train/instances_gripper_fg_bg.json"
        },
        "hsr_gripper_fg_bg_val":{
            "img_dir": "hsr/gripper_fg_bg/val/JPEGImages",
            "ann_file": "hsr/gripper_fg_bg/val/instances_gripper_fg_bg.json"
        },
        "hsr_gripper_fg_bg_test":{
            "img_dir": "hsr/gripper_fg_bg/test/JPEGImages",
            "ann_file": "hsr/gripper_fg_bg/test/instances_gripper_fg_bg.json"
        },       
        "hsr_obj_aug_gt_train":{
            "img_dir": "hsr/in_hand_aug/JPEGImages/ground-truth",
            "ann_file": "hsr/in_hand_aug/Annotations/instances_in_hand_augmented_ground_truth.json"
        },
        "hsr_obj_aug_maskrcnn_train":{
            "img_dir": "hsr/in_hand_aug/JPEGImages/maskrcnn",
            "ann_file": "hsr/in_hand_aug/Annotations/instances_in_hand_augmented_maskrcnn.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        }
    }

    @staticmethod
    def get(name):
        if "coco" in name or "lvis" in name or "hsr" in name or "srn" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
