import glob, os
import random
from PIL import Image
import numpy as np

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

