# --- 6) Hàm apply_chromosome ---
def apply_chromosome(image, chromo):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    out = image.copy()  # Đầu vào đã trong [0, 1]

    for sub in chromo.transformations:
        mask = sub["indices_mask"].astype(bool)
        if sub["activate_dropout"]:
            dp = np.random.rand(*out.shape[:2]) < sub["dropout_rate"]
            out[mask & dp] = out.min()
        if sub["activate_gaussian"]:
            noise = np.random.normal(0, sub["gaussian_sigma"] / 255.0, out.shape)
            out[mask] += noise[mask]
            out = np.clip(out, 0, 1)
        if sub["activate_brightness"]:
            out[mask] += sub["brightness_shift"]
            out = np.clip(out, 0, 1)
        if sub["activate_channel_shift"]:
            for c in range(3):
                out[:, :, c][mask] += sub["channel_shift_values"][c]
            out = np.clip(out, 0, 1)
    return out