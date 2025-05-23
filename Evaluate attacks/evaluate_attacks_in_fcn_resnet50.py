
from scipy.stats import mode

# loader_test: DataLoader cho val_dataset (clean)
loader_test = DataLoader(val_dataset, batch_size=2, shuffle=False)

# apply_chromosome(img_np, best_chromo)
# compute_iou(pred_mask, true_mask)
# compute_psnr(x_np, xadv_np, data_range=1.0)

# ---  Tạo wrapper cho mô hình phân đoạn ---
class SegmentationModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)['out']
        out = out.mean(dim=[2, 3])  # [batch_size, num_classes]
        return out

# Bao bọc modelA để trả về tensor phù hợp với torchattacks
wrapped_model = SegmentationModelWrapper(modelA)

# Khởi tạo các tấn công với wrapped_model
fgsm = torchattacks.FGSM(wrapped_model, eps=0.07)
pgd_10 = torchattacks.PGD(wrapped_model, eps=0.07, alpha=0.01, steps=10)  # PGD với 10 bước
pgd_100 = torchattacks.PGD(wrapped_model, eps=0.07, alpha=0.01, steps=100)  # PGD với 100 bước
deepfool = torchattacks.DeepFool(wrapped_model)
cw = torchattacks.CW(wrapped_model, c=1e-4, kappa=0, steps=1000, lr=1e-5)

def psnr_tensor(x, x_adv):
    x_np = x.cpu().numpy().transpose(1, 2, 0)
    xadv_np = x_adv.cpu().numpy().transpose(1, 2, 0)
    return compute_psnr(x_np, xadv_np, data_range=1.0)

def segrmt_attack(x, best_chromo):
    x_np = x.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    out = []
    for img in x_np:
        p = apply_chromosome(img, best_chromo)  # HxWxC
        out.append(torch.from_numpy(p.transpose(2, 0, 1)).float())
    return torch.stack(out).to(x.device)

def convert_segmentation_labels_to_classification(y):
    y_class = []
    for mask in y:
        mask_flat = mask.view(-1)  # Flatten mask
        values, counts = torch.unique(mask_flat, return_counts=True)
        if len(values) > 0:
            label = values[torch.argmax(counts)]
        else:
            label = 0  # Giá trị mặc định nếu mask rỗng
        y_class.append(label.item())
    return torch.tensor(y_class, dtype=torch.long).to(y.device)

def make_adv(atk, x, y, best_chromo=None):
    y_class = convert_segmentation_labels_to_classification(y)
    if atk == 'FGSM':
        return fgsm(x, y_class)
    if atk == 'PGD-10':
        return pgd_10(x, y_class)  # Sử dụng pgd_10
    if atk == 'PGD-100':
        return pgd_100(x, y_class)  # Sử dụng pgd_100
    if atk == 'DeepFool':
        return deepfool(x, y_class)
    if atk == 'C&W':
        return cw(x, y_class)
    if atk == 'SegRMT':
        return segrmt_attack(x, best_chromo)
    raise ValueError(atk)

# ---  Tính IoU gốc (clean IoU) ---
clean_iou_list = []
for x, y in loader_test:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = modelA(x)['out']
        preds = out.argmax(1).cpu().numpy()
    for p, t in zip(preds, y.cpu().numpy()):
        clean_iou_list.append(compute_iou(p, t))
clean_iou = np.mean(clean_iou_list)
print(f"Clean IoU = {clean_iou:.4f}")

# ---  Đánh giá mỗi tấn công ---
attacks = ['FGSM', 'PGD-10', 'PGD-100', 'DeepFool', 'C&W', 'SegRMT']
res = {}
for atk in attacks:
    psnrs, adv_ious = [], []
    for x, y in loader_test:
        x, y = x.to(device), y.to(device)
        x_adv = make_adv(atk, x, y, best_chromo)
        # PSNR per-sample
        for xi, xai in zip(x, x_adv):
            psnrs.append(psnr_tensor(xi, xai))
        # IoU on adv batch
        with torch.no_grad():
            out = modelA(x_adv)['out']
            preds = out.argmax(1).cpu().numpy()
        for p, t in zip(preds, y.cpu().numpy()):
            adv_ious.append(compute_iou(p, t))
    res[atk] = {
        'PSNR': np.mean(psnrs),
        'AdvIoU': np.mean(adv_ious),
        'ΔIoU': clean_iou - np.mean(adv_ious)
    }

# ---  In kết quả và vẽ bar-chart ---
print(f"{'Attack':<10} {'PSNR':>6}  {'AdvIoU':>7}  {'ΔIoU':>6}")
for atk, v in res.items():
    print(f"{atk:<10} {v['PSNR']:6.2f}   {v['AdvIoU']:6.3f}   {v['ΔIoU']:6.3f}")

# Bar chart ΔIoU
plt.figure(figsize=(6, 4))
plt.bar(res.keys(), [res[a]['ΔIoU'] for a in attacks])
plt.ylabel('ΔIoU')
plt.title('Robustness (ΔIoU) Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()