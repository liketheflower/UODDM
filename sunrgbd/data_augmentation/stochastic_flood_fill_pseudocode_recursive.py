EXPLORED, UNEXPLORED = 1, -1
RGB, DHS, NOT_RGB_NOT_DHS = 0, 1, -1
connect_probs = {RGB: 0.5, DHS: 0.5}
neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def opposite_image_type(img_type):
    return RGB if img_type == DHS else DHS


def stochastic_flood_fill(i, j, status, mask, curr_img_type):
    if (i, j) is not valid or status[i][j] == EXPLORED:return
    fill_this_location = False
    if mask[i][j] == curr_img_type:  # if same image type, mask it
        fill_this_location = True
    elif mask[i][j] == NOT_RGB_NOT_DHS:
        if random_number <= connect_probs[curr_img_type]:  # stochastic part
            fill_this_location = True
        else:
            mask[i][j] = opposite_image_type(curr_img_type)

    if fill_this_location:
        mask[i][j], status[i][j] == curr_img_type, EXPLORED
        for di, dj in neighbors:
            stochastic_flood_fill(i + di, j + dj, status, mask, curr_img_type)


def get_stochastic_flood_fill_mask(img_height, img_width):
    H, W = img_height, img_width
    status = UNEXPLORED * np.ones((H, W))
    mask = NOT_RGB_NOT_DHS * np.ones((H, W))
    mask[0][0] = RGB  # use first pixel as RGB for example
    for i in range(H):
        for j in range(W):
            if status[i][j] == EXPLORED:continue
            curr_img_type = mask[i][j]
            status, mask = stochastic_flood_fill(i, j, status, mask, curr_img_type)
    return mask
