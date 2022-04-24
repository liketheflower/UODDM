"""
Based on  convert_sunrgbd_bothrgbdhs_to_coco_style_train_only.py
add the stochastic flood fill inter modality mixture data augmentation
"""
import os, glob, json
import mmcv
import numpy as np
import random
import collections
from convert_sunrgbd_bothrgbdhs_to_coco_style_train_only import (
    sunrgbd80_classsnames_ids_list,
    get_categories_info,
)

info = {
    "description": "SUN RGBD coco styple RGB DHS images with sfl",
    "url": "",
    "version": "1.0",
    "year": 2022,
    "contributor": "Sun RGBD dataset creators, computer vision lab, Hunter College, CUNY",
    "date_created": "2022/01/01",
}


def get_images_info(img_files):
    """
    The image file has the format of "007663_dhs.png". For the coco style
    [
     {
        "license": 4, # we don't need this one
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
     },
     ...
    ]
    """
    assert len(img_files) > 0, "Image files are empty"
    cnt = collections.defaultdict(int)
    image_ids = []

    def is_augmented_img(img_fn):
        """
        not augmented img name format: 008546_rgb.png
        aug_id begins from 1
        augmented_img_name_format: 008546_rgbdhssff_<first pixel rgb or dhs>_<rgb extend p>_<dhs_extend p>_<aug_id>_.png
        """
        img_fn_split = img_fn.split("_")
        if len(img_fn_split) == 2:
            return False, 0
        #  005051_rgbdhssff_firstpixelrgb_082_018_001_.png, here 001 is aug id
        aug_id = int(img_fn_split[-2])
        assert aug_id >= 1, "aug id is invalid"
        return True, aug_id

    def extract_info_(img_fn):
        file_name = img_fn.split("/")[-1]
        img_id = int(file_name.split("_")[0])
        # Add delta to distinguish RGB with DHS
        img_id_delta = 1000000 if "dhs" in file_name else 0
        img_id += img_id_delta
        _, aug_id = is_augmented_img(file_name)
        # For each image, we can not have more than 1000 agumented image. If we do,
        # we need increase 1000.
        img_id = img_id * 1000 + aug_id
        image_ids.append(img_id)
        cnt[img_id_delta] += 1
        image = mmcv.imread(img_fn)
        height, width = image.shape[:2]
        # print("image height, width", height, width)
        return {"file_name": file_name, "height": height, "width": width, "id": img_id}

    ret = [extract_info_(img_fn) for img_fn in img_files]
    print("img delta in img info", cnt)
    assert len(image_ids) == len(set(image_ids)), "Duplicated image id detected"
    return ret


def get_annotations_info(img_files, label_folder):
    """
    the coco style annotation has the following format. The annotation contains all the
    objects from the images. The id is the annotation id. We set it as image_id * 100 +
    the object index within this image. Segmentation will use the same as the bbox to
    generate a FAKE image segmentation. In the coco style, segmentation list of vertices
      (x, y pixel positions). COCO bounding box format is
      [top left x position, top left y position, width, height].
    [
     {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768
     },
     ...
    ]
    """
    assert len(img_files) > 0, "Image files are empty"

    cnt = collections.defaultdict(int)

    def extract_label_info_(img_fn):
        file_name = img_fn.split("/")[-1]
        img_id = int(file_name.split("_")[0])
        bbox_fn = (
            label_folder
            + "sunrgbd80_bbox/"
            + str(img_id).zfill(6)
            + "_gt_unnormalized_xmin_y_min_w_hs_sunrgbd80.npy"
        )
        label_fn = (
            label_folder
            + "sunrgbd80_bbox/"
            + str(img_id).zfill(6)
            + "_gt_class_ids_sunrgbd80.npy"
        )
        bboxes, labels = np.load(bbox_fn), np.load(label_fn)
        img_id_delta = 100000 if "dhs" in file_name else 0
        cnt[img_id_delta] += 1
        img_id += img_id_delta
        if bboxes.size == 0:
            return []
        bboxes = bboxes.tolist()
        ret = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            category_id = int(labels[i])
            # add point x1 y1 x2 y2 x1 y1 to avoid the problem mentioned here
            # https://github.com/cocodataset/cocoapi/issues/139
            segmentation = [bbox + bbox[:2]]
            area = bbox[2] * bbox[3]
            anno_id = img_id * 100 + i
            ret.append(
                {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": anno_id,
                }
            )
        return ret

    ret = []
    for fn in img_files:
        ret += extract_label_info_(fn)
    print("img delta in img annotation", cnt)
    return ret


def generate_annotation(img_folder, label_folder, annotation_filename):
    """
    COCO annotation json file has the following keys
    dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    we only need generate "images" and "annotations"
    The creation follows this article
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    """
    img_files = sorted(glob.glob(img_folder + "*.png"))
    random.seed(42)
    random.shuffle(img_files)
    print(f"len of img_files {len(img_files)}")
    images_info = get_images_info(img_files)
    annotations_info = get_annotations_info(img_files, label_folder)
    categories_info = get_categories_info()
    final_annotations = {
        "info": info,
        "licenses": "none",
        "images": images_info,
        "annotations": annotations_info,
        "categories": categories_info,
    }
    fn = label_folder + annotation_filename
    with open(fn, "w") as f:
        json.dump(final_annotations, f)


def convert_data_to_coco_style(img_folder, label_folder):
    """
    Convert our SUN RGBD dataset to coco style
    """
    # For train
    generate_annotation(img_folder, label_folder, "det_train_rgb_and_dhs_with_sfl.json")
    # For test should not be generated as it already has
    """
    img_folder = dataset_folder + "val/"+ img_type + "/"      
    label_folder = dataset_folder + "val/gts/raw_gts/"
    generate_annotation(img_folder, label_folder, "det_val"+img_type + ".json")
    """


if __name__ == "__main__":
    img_folder = (
        "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/train/rgbdhs_sff/"
    )
    label_folder = (
        "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/train/gts/raw_gts/"
    )
    convert_data_to_coco_style(img_folder, label_folder)
