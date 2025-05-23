# --- 4) Hàm apply_distortions ---
def apply_distortions(image, spatial_rate, channel_dropout):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_copy = image * std + mean

    h, w, c = img_copy.shape
    num_pixels = int(spatial_rate * h * w)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    img_copy[ys, xs, :] = 0

    for idx, dropout in enumerate(channel_dropout):
        if dropout:
            img_copy[:, :, idx] = 0

    img_copy = (img_copy - mean) / std
    return img_copy