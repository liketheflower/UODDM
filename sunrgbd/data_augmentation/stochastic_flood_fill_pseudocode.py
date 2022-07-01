EXPLORED, UNEXPLORED = 100,  -1
NOT_RGB_NOT_DHS = -1
neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
RGB, DHS = 0, 1

def opposite_image_type(img_type):
    return RGB if img_type == DHS else DHS


def get_connect_prob(image_type, rgb_connect_prob=0.5, dhs_connect_prob=0.5):
    return rgb_connect_prob if image_type == RGB else dhs_connect_prob


def stochastic_flood_fill(i, j, status, mixture_mask, connect_prob, current_image_type):
    open_list = [(i, j)]
    # use iterative way instead of recursive to avoid stack overflow
    while open_list:
        i, j = open_list.pop()
        for di, dj in neighbors:
            r, c = i + di, j + dj
            if (r, c) is not valid or status[r][c] == EXPLORED:
                continue
            if status[r][c] == current_img_type:
                status[r][c] = EXPLORED; mixture_mask[r][c] = current_img_type
                open_list.append((r, c))
                continue
            # Edge is connected or not. If not connected, then the other side will
            # be set as opposite mask
            if random_number <= connect_prob:
                status[r][c] = EXPLORED; mixture_mask[r][c] = current_img_type
                open_list.append((r, c))
            else:
                mixture_mask[r][c] = opposite_image_type(current_img_type)
    return status, mixture_mask


def get_stochastic_flood_fill_mask(img_height, img_width):
    H, W = img_height, img_width
    status = UNEXPLORED * np.ones((H, W))
    mixture_mask = NOT_RGB_NOT_DHS * np.ones((H, W))
    # Use first pixel as RGB for example
    mixture_mask[0][0] = RGB
    for i in range(H):
        for j in range(W):
            if status[i][j] == EXPLORED:continue
            current_image_type = mixture_mask[i][j]
            connect_prob = get_connect_prob(current_image_type)
            status, mixture_mask = stochastic_flood_fill(
                i, j, status, mixture_mask, connect_prob, current_image_type
            )
    return mixture_mask
