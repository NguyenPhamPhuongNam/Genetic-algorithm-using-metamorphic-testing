def get_samples_from_loader(loader, n_samples):
    samples = []
    iterator = iter(loader)
    for _ in range(n_samples):
        try:
            sample_img, sample_mask = next(iterator)
            samples.append((sample_img, sample_mask))
        except StopIteration:
            iterator = iter(loader)
            sample_img, sample_mask = next(iterator)
            samples.append((sample_img, sample_mask))
    return samples

# Lấy 3 samples từ loader
samples = get_samples_from_loader(loader, 3)

# ---  Khởi tạo GA ---
ga = GeneticAlgorithm(
    img_shape=(512, 800, 3),
    seg_model=seg_model,
    compute_iou=compute_iou,
    pop_size=10,
    generations=9,
    crossover_method="two_point",
    alpha=0.5,
    beta=2.0
)

# --- Hàm hiển thị ảnh gốc và ảnh perturbed ---
def generate_and_show_image(img_tensor, chromosome, idx):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = img_tensor[0].numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    perturbed_img = apply_chromosome(img_np, chromosome)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img_np)
    axs[0].set_title(f"Original Image {idx+1}")
    axs[0].axis('off')

    axs[1].imshow(perturbed_img)
    axs[1].set_title(f"Perturbed Image {idx+1}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# ---  Lặp qua các samples và chạy GA ---
for idx, (sample_img, sample_mask) in enumerate(samples):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = sample_img[0].numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    mask_np = sample_mask[0].numpy()

    best = ga.run(img_np, mask_np)
    best_chromosome = best

    print(f"\nBest Chromosome found for sample {idx+1}:")
    print("Best length:", len(best.transformations))
    for i, sub in enumerate(best_chromosome.transformations, 1):
        print(f"\n Sub-vector #{i}:")
        print(f"   • Dropout:       active={sub['activate_dropout']}, rate={sub['dropout_rate']:.3f}")
        print(f"   • Gaussian:      active={sub['activate_gaussian']}, σ={sub['gaussian_sigma']:.2f}")
        print(f"   • Brightness:    active={sub['activate_brightness']}, shift={sub['brightness_shift']:.3f}")
        print(f"   • Channel Shift: active={sub['activate_channel_shift']}, shifts={sub['channel_shift_values']}")
        total_pixels = sub['indices_mask'].size
        affected = int(np.sum(sub['indices_mask']))
        print(f"   • Pixels affected mask: {affected} / {total_pixels}")

    generate_and_show_image(sample_img, best_chromosome, idx)