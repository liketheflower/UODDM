import glob, os
import random
from stochastic_flood_fill import stochastic_flood_fill_for_an_image

biased_week_bounds = [0.1, 0.2]
biased_strong_bounds = [0.8, 0.9]
unbiased_bounds = [0.1, 0.9]
EPS = 0.0000000001
def inter_modality_mixture(rgb_filename: str, dhs_filename: str, patch_size: int, save_path: str, save_filename: str,first_patch:str = "rgb"):
    """
    Implement the inter modality mixture.
    Parameters:
       rgb_filename: the file name of the RGB image.
       dhs_filename: the filename of the DHS image.
       patch_size: the size of each square patch. For simplicity, we use square instead of rectangle, so it only need one parameter.
       first_patch: whether the first patch is from rgb or dhs. We should support both begin with rgb or dhs.
       save_path: the path used to save the image after the inter modality mixture
       save_filename: The name used to save the mixture image.
    Return:
      None as the generated file is saved.
    """

    
    dhs_frame = Image.open(dhs_filename)
    rgb_frame = Image.open(rgb_filename)
    
    h,w = rgb_frame.size
    mixture_frame = np.zeros((w, h, 3),dtype=int)
    rgb = np.array(rgb_frame)
    dhs = np.array(dhs_frame)
    c = 3 # three channels
    w_step = int(w/patch_size)
    h_step = int(h/patch_size)

    if first_patch=="rgb":    
        for i in range(c):
            for j in range(0,patch_size,2):
                for k in range(patch_size):
                    if k%2==0:
                        
                        mixture_frame[j*(w_step) : (j+1)*(w_step), k*(h_step) : (k+1)*(h_step), i] = rgb[j*(w_step) : (j+1)*(w_step), k*(h_step) : (k+1)*(h_step), i]
                        
                    else:  
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = dhs[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
            
            for j in range(1,patch_size,2):
                for k in range(patch_size):
                    if k%2==0:
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = dhs[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
                    else:  
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = rgb[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]


    if first_patch=="dhs":
        
        for i in range(c):
            for j in range(0,patch_size,2):
                for k in range(patch_size):
                    if k%2==0:
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = dhs[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
                    else:  
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = rgb[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
            
            for j in range(1,patch_size,2):
                for k in range(patch_size):
                    if k%2==0:
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = rgb[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
                    else:  
                        mixture_frame[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i] = dhs[j*(w_step):(j+1)*(w_step), k*(h_step):(k+1)*(h_step), i]
    
    im = Image.fromarray(mixture_frame.astype(np.uint8))
    
    im.save(os.path.join(save_path,save_filename))  

def get_random_number(lower, higher):
    while True:
        random_num = random.random()
        if random_num >= lower and random_num <= higher:
            return random_num


def get_fn_prefix(fn):
    pure_fn = fn.split("/")[-1]
    return pure_fn.split("_")[0]


def aug_this_image(rgb_fn, dhs_fn, connect_probs, save_path, cnt):
    p1, p2 = connect_probs
    first_pixel = "firstpixelrgb"
    if random.random() <= 0.5:
        # Set it to 1.0 to force use RGB
        rgb_prob_of_first_pixel = 1.0
    else:
        first_pixel = "firstpixeldhs"
        rgb_prob_of_first_pixel = 0.0
    save_fn = "_".join(
        [
            first_pixel,
            str(int(p1 * 100)).zfill(3),
            str(int(p2 * 100)).zfill(3),
            str(cnt).zfill(3),
        ]
    )
    save_fn = get_fn_prefix(rgb_fn) + "_rgbdhssff_" + save_fn + "_.png"
    stochastic_flood_fill_for_an_image(
        rgb_fn, dhs_fn, rgb_prob_of_first_pixel, connect_probs, save_path, save_fn
    )
    return cnt + 1


def aug_all_images(img_path, biased_aug_num=2, unbiased_aug_num=2):
    rgb_img_fns = sorted(glob.glob(img_path + "*_rgb.png"))
    dhs_img_fns = [img_path + get_fn_prefix(fn) + "_dhs.png" for fn in rgb_img_fns]
    assert all(os.path.isfile(fn) for fn in dhs_img_fns)
    save_path = img_path
    for rgb_fn, dhs_fn in zip(rgb_img_fns, dhs_img_fns):
        assert get_fn_prefix(rgb_fn) == get_fn_prefix(dhs_fn), "Not matched filename"
        cnt = 1
        for i in range(biased_aug_num):
            week_p = get_random_number(*biased_week_bounds)
            strong_p = get_random_number(*biased_strong_bounds)
            assert (
                week_p >= biased_week_bounds[0] - EPS
                and week_p <= biased_week_bounds[1] + EPS
            ), """Wrong random number"""
            assert (
                strong_p >= biased_strong_bounds[0] - EPS
                and strong_p <= biased_strong_bounds[1] + EPS
            ), """Wrong random number"""
            connect_probs = [strong_p, week_p]
            if random.random() <= 0.5:  # flip
                connect_probs = connect_probs[::-1]
            cnt = aug_this_image(rgb_fn, dhs_fn, connect_probs, save_path, cnt)
            print(rgb_fn, dhs_fn, cnt)
        for i in range(unbiased_aug_num):
            p1 = get_random_number(*unbiased_bounds)
            p2 = get_random_number(*unbiased_bounds)
            assert (
                p1 >= unbiased_bounds[0] - EPS and p1 <= unbiased_bounds[1] + EPS
            ), """Wrong random number"""
            assert (
                p2 >= unbiased_bounds[0] - EPS and p2 <= unbiased_bounds[1] + EPS
            ), """Wrong random number"""
            connect_probs = [p1, p2]
            cnt = aug_this_image(rgb_fn, dhs_fn, connect_probs, save_path, cnt)
            print(rgb_fn, dhs_fn, cnt)


if __name__ == "__main__":
    img_path = (
        "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/train/rgbdhs_sff/"
    )
    aug_all_images(img_path)
