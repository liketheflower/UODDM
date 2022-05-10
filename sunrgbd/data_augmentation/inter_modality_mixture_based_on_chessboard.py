import glob, os
from PIL import Image
import numpy as np
from chessboard_mixture import chessboard_mixture


def get_fn_prefix(fn):
    pure_fn = fn.split("/")[-1]
    return pure_fn.split("_")[0]


def aug_this_image(rgb_fn, dhs_fn, patch_size, save_path, cnt):
    save_fn = get_fn_prefix(rgb_fn) + "_rgbdhchessboard1_" + str(cnt).zfill(3) + "_.png"
    chessboard_mixture(
        rgb_filename=rgb_fn,
        dhs_filename=dhs_fn,
        patch_size=patch_size,
        first_patch="rgb",
        save_filename=save_path + save_fn,
    )
    return cnt + 1


def aug_all_images(img_path, patch_size=1):
    rgb_img_fns = sorted(glob.glob(img_path + "*_rgb.png"))
    dhs_img_fns = [img_path + get_fn_prefix(fn) + "_dhs.png" for fn in rgb_img_fns]
    assert all(os.path.isfile(fn) for fn in dhs_img_fns)
    save_path = img_path
    for rgb_fn, dhs_fn in zip(rgb_img_fns, dhs_img_fns):
        assert get_fn_prefix(rgb_fn) == get_fn_prefix(dhs_fn), "Not matched filename"
        aug_this_image(rgb_fn, dhs_fn, patch_size, save_path, 1)


if __name__ == "__main__":
    img_path = "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/train/rgbdhs_chess1/"
    aug_all_images(img_path)
