# --- 8) Hàm evaluate_fitness ---
def evaluate_fitness(img, mask, chromo, seg_model, alpha=1.0, beta=1.0):
    pert = apply_chromosome(img, chromo)
    psnr = compute_psnr(img, pert, data_range=1.0)
    if psnr < 20:
        chromo.fitness = -np.inf
        return
    pred = seg_model.predict(pert)
    iou = compute_iou(pred, mask)
    chromo.fitness = alpha * psnr - beta * iou