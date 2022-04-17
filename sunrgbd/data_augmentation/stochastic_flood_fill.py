import cv2
import os
import numpy as np
import random

orange = [244, 109, 67]
green = [102, 189, 99]
gray = [135, 135, 135]
colors = np.array([orange[::-1], gray[::-1]])

# this number will be added by 0 or 1 so we will have
# 100 for one side and 101 for the other
EXPLORED = 100
UNEXPLORED = -1
# this number will be added by 0 or 1 so we will have
# 0 for one side and 1 for the other
FRONTIER = 0
FOUR_WAYS = True

if FOUR_WAYS:
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
else:
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]


def stochastic_flood_fill(grid, i, j, random_numbers, random_num_idx, connect_probs):
    """
    Parameters:
        grid: a shape of H by W np array. It is a status array to indicate whether each
          cell is explored, unexplored or the frontier. Initially, the grid[0][0] is the
          frontier and it will be initialized to either 0 or 1. All the rest are unxplored.
        i, j: the start point of this exploration.
        random_numbers: pregenerating random numbers so the process can be reproduced if
          we use fixed random seed. In production, this can be replaced by generating
          random number on the fly.
       random_num_idx: the next index to be used for pre generated random number.
       connect_probs: the probability of the edge to to unexplored neighbor can be
        connected for side 0 and 1. It has two numbers one is for side 0 and one is for
        side 1, they are not necessary to be sum to 1. For example, each side can both
        have a large number such as 0.7 to explore the unexplored space. Also, we can
        set one is 0.8 the other is 0.4 with bias to side 0.
    """
    R, C = len(grid), len(grid[0])
    assert 0 <= i < R and 0 <= j < C, f"invalid i {i} or j {j}"
    open_list = [(i, j)]
    begin_status = int(grid[i][j])
    # use iterative way instead of recursive to avoid stack overflow
    while open_list:
        i, j = open_list.pop()
        for di, dj in dirs:
            r, c = i + di, j + dj
            if (
                r < 0
                or r >= R
                or c <= 0
                or c >= C
                or (int(grid[r][c]) not in [UNEXPLORED, begin_status])
            ):
                continue
            if int(grid[r][c]) == begin_status:
                grid[r][c] = EXPLORED + begin_status
                open_list.append((r, c))
                continue
            # edge is connected or not. If not connected, then the other side will
            # be set as opposite number
            if random_numbers[random_num_idx] <= connect_probs[begin_status]:
                grid[r][c] = EXPLORED + begin_status
                open_list.append((r, c))
            else:
                # Flip the status
                grid[r][c] = 1 - begin_status
            random_num_idx += 1
    return grid, random_num_idx


def get_stochastic_flood_fill_mask(
    rgb_prob_of_first_pixel, img_height, img_width, connect_probs, random_seed=None
):
    """
    Parameters:
       rgb_prob_of_first_pixel: the probability of the first pixel is rgb pixel
       connect_probs: the probability of the edge to to unexplored neighbor can be
        connected for side 0 and 1. It has two numbers one is for side 0 and one is for
        side 1, they are not necessary to be sum to 1. For example, each side can both
        have a large number such as 0.7 to explore the unexplored space. Also, we can
        set one is 0.8 the other is 0.4 with bias to side 0.
    """
    H, W = img_height, img_width
    mask = -1 * np.ones((H, W))
    # Using the 0 as rgb image and 1 as the DHS image.
    if random.random() <= rgb_prob_of_first_pixel:
        mask[0][0] = 0
    else:
        mask[0][0] = 1
    if random_seed is not None:
        random.seed(random_seed)
    DELTA = 2  # generate extra random numbers
    random_numbers = [random.random() for _ in range(H * W * (len(dirs) + DELTA))]
    print(random_numbers[:10])
    random_num_idx = 0
    for i in range(H):
        for j in range(W):
            if int(mask[i][j]) in [0, 1]:
                mask, random_num_idx = stochastic_flood_fill(
                    mask, i, j, random_numbers, random_num_idx, connect_probs
                )
    mask = mask % 2
    return mask

def stochastic_flood_fill_for_a_dummy_image(
    rgb_prob_of_first_pixel,
    connect_probs,
    save_path="/Users/jimmy/Downloads/stochastic_flood_fill"
    + FOUR_WAYS * "four_way"
    + "/",
):
    """
    Test a dummy image, use color[0] as type A image and color[1] as type B image.
    Parameters:
       rgb_prob_of_first_pixel: the probability of the first pixel is rgb pixel
       connect_probs: the probability of the edge to to unexplored neighbor can be
        connected for side 0 and 1. It has two numbers one is for side 0 and one is for
        side 1, they are not necessary to be sum to 1. For example, each side can both
        have a large number such as 0.7 to explore the unexplored space. Also, we can
        set one is 0.8 the other is 0.4 with bias to side 0.
    """
    os.makedirs(save_path, exist_ok=True)
    H, W = 100, 200
    mask = get_stochastic_flood_fill_mask(1.0, H, W, connect_probs, 42)
    img = np.zeros((H, W, 3))
    img[mask == 0] = colors[0]
    img[mask == 1] = colors[1]
    save_fn = (
        FOUR_WAYS * "four_ways_"
        + "stochastic_flood_fill_demo"
        + "_".join([str(p)[:4] for p in connect_probs])
        + ".png"
    )
    cv2.imwrite(save_path + save_fn, img)


def stochastic_flood_fill_for_an_image(
    rgb_filename: str,
    dhs_filename: str,
    rgb_prob_of_first_pixel,
    connect_probs,
    save_path="/Users/jimmy/Downloads/stochastic_flood_fill"
    + FOUR_WAYS * "four_way_real"
    + "/",
    save_fn=None,
):
    """
    Test on ral images. mask with value 0 is from rgb image, with value 1 is from dhs
      image.
    Parameters:
       rgb_prob_of_first_pixel: the probability of the first pixel is rgb pixel
       connect_probs: the probability of the edge to to unexplored neighbor can be
        connected for side 0 and 1. It has two numbers one is for side 0 and one is for
        side 1, they are not necessary to be sum to 1. For example, each side can both
        have a large number such as 0.7 to explore the unexplored space. Also, we can
        set one is 0.8 the other is 0.4 with bias to side 0.
    """
    os.makedirs(save_path, exist_ok=True)
    rgb = cv2.imread(rgb_filename)
    dhs = cv2.imread(dhs_filename)
    assert rgb.shape == dhs.shape, "rgb and dhs images shape should match!"
    H, W = rgb.shape[:2]
    # rgb_prob_of_first_pixel set as 0.5 so rgb and dhs has equal opportunity
    mask = get_stochastic_flood_fill_mask(rgb_prob_of_first_pixel, H, W, connect_probs)
    img = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            selected_img = rgb if int(mask[i, j]) == 1 else dhs
            img[i, j] = selected_img[i, j]
    save_fn = (
        FOUR_WAYS * "four_ways_"
        + "stochastic_flood_fill_demo"
        + "_".join([str(p)[:4] for p in connect_probs])
        + ".png"
    )
    cv2.imwrite(save_path + save_fn, img)


if __name__ == "__main__":
    rgb_filename = "./../demo_images/006223_rgb.png"
    dhs_filename = "./../demo_images/006223_dhs.png"
    for prob in [
        (a, b) for a in np.arange(0.1, 1.0, 0.1) for b in np.arange(0.1, 1.0, 0.1)
    ]:
        stochastic_flood_fill_for_a_dummy_image(1, prob)
        save_path = "/Users/jimmy/Downloads/stochastic_flood_fill_real_fourway/"
        save_fn = (
            "stochastic_flood_fill_real_"
            + "_".join([str(p)[:4] for p in prob])
            + ".png"
        )
        stochastic_flood_fill_for_an_image(
            rgb_filename, dhs_filename, 1.0, prob, save_path, save_fn
        )
