# --- 7) Hàm compute_iou ---
def compute_iou(pred, true):
    intersection = np.logical_and(pred == true, true != -1).sum()
    union = np.logical_or(pred != -1, true != -1).sum()
    return intersection / (union + 1e-9)