loader_test = DataLoader(val_dataset, batch_size=8, shuffle=False)
# ---  Device, DataLoader test ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# loader_test: DataLoader cho val_dataset (CIFAR10Segmentation), normalize đã được áp
# e.g.
# loader_test = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ---  Load two models --

modelA.load_state_dict(torch.load("deeplabv3_cifar10.pth", map_location=device, weights_only=True))
modelA.eval()

model.load_state_dict(torch.load("new_model_on_perturbed.pth", map_location=device, weights_only=True))
model.eval()

# Hàm evaluate transfer: sinh adv trên A, đánh giá IoU trên B
def eval_transfer(atk):
    ious = []
    for x, y in loader_test:
        x, y = x.to(device), y.to(device)
        # 1) Sinh adversarial trên Model A
        x_adv = make_adv(atk, x, y)
        # 2) Dự đoán trên Model B
        with torch.no_grad():
            outB = model(x_adv)['out']
        predsB = outB.argmax(1).cpu().numpy()
        # 3) Tính IoU trên từng ảnh
        for pb, tb in zip(predsB, y.cpu().numpy()):
            ious.append(compute_iou(pb, tb))
    return np.mean(ious)

# Clean IoU trên Model B
clean_ious_B = []
for x, y in loader_test:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        outB = model(x)['out']
    predsB = outB.argmax(1).cpu().numpy()
    for pb, tb in zip(predsB, y.cpu().numpy()):
        clean_ious_B.append(compute_iou(pb, tb))
clean_iou_B = np.mean(clean_ious_B)
print(f"Clean IoU on Model B = {clean_iou_B:.4f}\n")

# Đánh giá transferability
print(f"{'Attack':<10} {'IoU_on_B':>8} {'ΔIoU':>8}")
for atk in attacks:
    adv_iou_B = eval_transfer(atk)
    print(f"{atk:<10} {adv_iou_B:8.3f} {clean_iou_B - adv_iou_B:8.3f}")