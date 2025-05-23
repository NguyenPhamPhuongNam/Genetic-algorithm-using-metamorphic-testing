# --- 4) Utility để tạo sub-vector ngẫu nhiên ---
def generate_random_subvector(img_shape):
    h, w, _ = img_shape
    return {
        "activate_dropout": random.choice([0, 1]),
        "activate_gaussian": random.choice([0, 1]),
        "activate_brightness": random.choice([0, 1]),
        "activate_channel_shift": random.choice([0, 1]),
        "dropout_rate": random.uniform(0.01, 0.1),
        "gaussian_sigma": random.uniform(1, 5),
        "brightness_shift": random.uniform(-0.2, 0.2),
        "channel_shift_values": [random.uniform(-0.2, 0.2) for _ in range(3)],
        "indices_mask": np.random.choice([0, 1], size=(h, w), p=[0.9, 0.1])
    }