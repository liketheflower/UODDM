import glob, os
import cv2
import random
from PIL import Image
import numpy as np


def chessboard_mixture(
    rgb_filename: str,
    dhs_filename: str,
    patch_size: int,
    first_patch: str = "rgb",
    save_filename: str = None,
):
    """
    Implement the inter modality mixture.
    Parameters:
       rgb_filename: the file name of the RGB image.
       dhs_filename: the filename of the DHS image.
       patch_size: the size of each square patch. For simplicity, we use square
         instead of rectangle, so it only need one parameter.
       first_patch: whether the first patch is from rgb or dhs. We should support both
          begin with rgb or dhs.
       save_filename: The name used to save the mixture image.
    Return:
      None as the generated file is saved.
    """
    rgb = cv2.imread(rgb_filename)
    dhs = cv2.imread(dhs_filename)
    assert first_patch in ["rgb", "dhs"], "Only can be rgb or dhs"
    assert rgb.shape == dhs.shape, "Shape of rgb and dhs are suppose to be matched"
    assert (
        len(rgb.shape) == 3 and rgb.shape[2] == 3
    ), """Shape should have 3 dimension                                                 
    and the last channel should be 3"""
    output_img = np.zeros(rgb.shape, dtype=rgb.dtype)
    images = [rgb, dhs]
    begin_idx = 0 if first_patch == "rgb" else 1
    H, W, _ = rgb.shape
    for i in range(0, H, patch_size):
        # For each row, Alterative pick up an image. For example, when begin_idx is 0,
        # for the first column, the first row, k will be 0 (rgb), for the second row
        # k will be 1, we use dhs.
        if (i // patch_size) % 2 == 0:
            k = begin_idx
        else:
            # begin with another image
            k = 1 - begin_idx
        for j in range(0, W, patch_size):
            # For each column, Alterative pick up an image.
            # For example, when begin_idx is 0,
            # for the first row, the first column will be 0(rgb), the second column
            # will be 1, we use dhs.
            k = (k + (j // patch_size)) % 2
            row_begin, row_end = i, min(i + patch_size, H)
            col_begin, col_end = j, min(j + patch_size, W)
            output_img[row_begin:row_end, col_begin:col_end] = images[k][
                row_begin:row_end, col_begin:col_end
            ]
    cv2.imwrite(save_path + save_filename, output_img)


def inter_modality_mixture(
    rgb_filename: str,
    dhs_filename: str,
    patch_size: int,
    save_path: str,
    save_filename: str,
    first_patch: str = "rgb",
):
    """
    Implement the inter modality mixture.
    Parameters:
       rgb_filename: the file name of the RGB image.
       dhs_filename: the filename of the DHS image.
       patch_size: the size of each square patch. For simplicity, we use square instead
         of rectangle, so it only need one parameter.
       first_patch: whether the first patch is from rgb or dhs. We should support both
         begin with rgb or dhs.
       save_path: the path used to save the image after the inter modality mixture
       save_filename: The name used to save the mixture image.
    Return:
      None as the generated file is saved.
    """
    dhs_frame = Image.open(dhs_filename)
    rgb_frame = Image.open(rgb_filename)

    h, w = rgb_frame.size
    mixture_frame = np.zeros((w, h, 3), dtype=int)
    rgb = np.array(rgb_frame)
    dhs = np.array(dhs_frame)
    c = 3  # three channels
    w_step = int(w / patch_size)
    h_step = int(h / patch_size)

    if first_patch == "rgb":
        for i in range(c):
            for j in range(0, patch_size, 2):
                for k in range(patch_size):
                    if k % 2 == 0:

                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = rgb[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]

                    else:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = dhs[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]

            for j in range(1, patch_size, 2):
                for k in range(patch_size):
                    if k % 2 == 0:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = dhs[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]
                    else:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = rgb[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]

    if first_patch == "dhs":

        for i in range(c):
            for j in range(0, patch_size, 2):
                for k in range(patch_size):
                    if k % 2 == 0:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = dhs[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]
                    else:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = rgb[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]

            for j in range(1, patch_size, 2):
                for k in range(patch_size):
                    if k % 2 == 0:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = rgb[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]
                    else:
                        mixture_frame[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ] = dhs[
                            j * (w_step) : (j + 1) * (w_step),
                            k * (h_step) : (k + 1) * (h_step),
                            i,
                        ]
    im = Image.fromarray(mixture_frame.astype(np.uint8))
    im.save(os.path.join(save_path, save_filename))


if __name__ == "__main__":
    rgb_filename = "./../demo_images/006223_rgb.png"
    dhs_filename = "./../demo_images/006223_dhs.png"
    save_path = "/Users/jimmy/Downloads/chessboard/"
    os.makedirs(save_path, exist_ok=True)
    for patch_size in range(1, 40, 2):
        save_fn = "chessboard" + str(patch_size).zfill(3) + ".png"
        chessboard_mixture(
            rgb_filename=rgb_filename,
            dhs_filename=dhs_filename,
            patch_size=patch_size,
            first_patch="rgb",
            save_fn=save_path + save_fn,
        )
