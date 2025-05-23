root_dir = '/kaggle/input/lyft-udacity-challenge'
base_dataset = LyftUdacityDataset(root_dir, split='train', transform=transform)
val_dataset = LyftUdacityDataset(root_dir, split='val', transform=transform)
normalize_only = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
best_chromo = best_chromosome  

#  Dataset perturbed cho cả train và val
class PerturbedLyftUdacity(Dataset):
    def __init__(self, base_ds, chromo):
        self.base = base_ds
        self.chromo = chromo

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        # Denormalize với mean và std của Lyft-Udacity
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        per_np = apply_chromosome(img_np, self.chromo)
        per_t = torch.from_numpy(per_np.transpose(2, 0, 1)).float()
        per_t = normalize_only(per_t)  # Chuẩn hóa lại
        return per_t, mask

#  Khởi tạo loaders
train_dataset = PerturbedLyftUdacity(base_dataset, best_chromo)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_clean_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
val_pert_loader = DataLoader(PerturbedLyftUdacity(val_dataset, best_chromo),
                             batch_size=1, shuffle=False)

#  Model, optimizer
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
modelA = deeplabv3_resnet50(weights="DEFAULT")  # Load trên CPU trước
modelA.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=(1, 1), stride=(1, 1))  # Chỉnh số class
modelA = modelA.to(device)  # Chỉ chuyển sang GPU sau khi đã sửa
model = fcn_resnet50(pretrained=False, num_classes=13).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.5,
#     patience=5,
#     threshold=1e-4,  # Ngưỡng cải thiện loss tối thiểu
#     verbose=True
# )



# # --- 5) Composite loss ---
# def composite_loss(model, x_clean, y_clean, x_adv, y_adv, alpha_1=1.0, beta_1=0.5):
#     lc = F.cross_entropy(model(x_clean)['out'], y_clean)
#     la = F.cross_entropy(model(x_adv)['out'], y_adv)
#     lr = F.mse_loss(model(x_clean)['out'], model(x_adv)['out'])
#     return lc + alpha_1 * la + beta_1 * lr

# # ---  Training + 2-way Validation ---
# epochs = 30
# train_losses = []
# val_losses_clean, val_losses_pert = [], []
# val_accs_clean, val_accs_pert = [], []

# for ep in range(1, epochs + 1):
#     # --- Train ---
#     model.train()
#     running = 0.0
#     for x_adv, masks in train_loader:
#         x_adv, masks = x_adv.to(device), masks.to(device)
#         optimizer.zero_grad()
#         loss = composite_loss(model, x_adv, masks, x_adv, masks, alpha_1=1.0, beta_1=0.5)
#         loss.backward()
#         optimizer.step()
#         running += loss.item()
    
#     train_loss = running / len(train_loader)
#     train_losses.append(train_loss)

#     # --- Val clean & perturbed ---
#     model.eval()
#     rc, rp = 0.0, 0.0  # Losses
#     cc, cp = 0, 0      # Correct
#     tc, tp = 0, 0      # Total

#     with torch.no_grad():
#         for (imgs_c, masks_c), (imgs_p, masks_p) in zip(val_clean_loader, val_pert_loader):
#             # Clean
#             imgs_c, masks_c = imgs_c.to(device), masks_c.to(device)
#             out_c = model(imgs_c)['out']
#             rc += F.cross_entropy(out_c, masks_c).item()
#             preds_c = out_c.argmax(1)
#             cc += (preds_c == masks_c).sum().item()
#             tc += masks_c.numel()

#             # Perturbed
#             imgs_p, masks_p = imgs_p.to(device), masks_p.to(device)
#             out_p = model(imgs_p)['out']
#             rp += F.cross_entropy(out_p, masks_p).item()
#             preds_p = out_p.argmax(1)
#             cp += (preds_p == masks_p).sum().item()
#             tp += masks_p.numel()

#     val_clean_loss = rc / len(val_clean_loader)
#     val_pert_loss = rp / len(val_pert_loader)
#     val_clean_acc = cc / tc
#     val_pert_acc = cp / tp

#     val_losses_clean.append(val_clean_loss)
#     val_losses_pert.append(val_pert_loss)
#     val_accs_clean.append(val_clean_acc)
#     val_accs_pert.append(val_pert_acc)

#     print(f"Epoch {ep}/{epochs}"
#           f" | Train: {train_loss:.4f}"
#           f" | Val Clean: {val_clean_loss:.4f}, Acc: {val_clean_acc:.4f}"
#           f" | Val Pert: {val_pert_loss:.4f}, Acc: {val_pert_acc:.4f}")

#     # Scheduler step dựa trên perturbed val loss
#     scheduler.step(val_pert_loss)

# # ---  Vẽ đồ thị ---
# plt.figure()
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses_clean, label="Val Clean Loss")
# plt.plot(val_losses_pert, label="Val Perturbed Loss")
# plt.legend()
# plt.title("Loss Curves")
# plt.show()

# plt.figure()
# plt.plot(val_accs_clean, label="Val Clean Acc")
# plt.plot(val_accs_pert, label="Val Perturbed Acc")
# plt.legend()
# plt.title("Accuracy")
# plt.show()

# # --- Lưu model ---

# torch.save(model.state_dict(), "new_model_on_perturbed.pth")
# torch.save(modelA.state_dict(), "deeplabv3_cifar10.pth")

# Theo dõi thời gian
start_time = time.time()
# Warm-up scheduler
warmup_epochs = 3
warmup_factor = 0.1
base_lr = 5e-4
warmup_lr = base_lr * warmup_factor

# Hàm điều chỉnh learning rate trong warm-up
def adjust_lr(optimizer, epoch, warmup_epochs, warmup_lr, base_lr):
    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * (epoch / warmup_epochs)
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Scheduler chính: Kết hợp CosineAnnealingLR và ReduceLROnPlateau
epochs = 7
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
# fallback_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.5,
#     patience=5,
#     threshold=1e-4,
#     verbose=True
# )

# --- 5) Composite loss ---
def composite_loss(model, x_clean, y_clean, x_adv, y_adv, alpha_1=1.0, beta_1=0.5):
    lc = F.cross_entropy(model(x_clean)['out'], y_clean)
    la = F.cross_entropy(model(x_adv)['out'], y_adv)
    lr = F.mse_loss(model(x_clean)['out'], model(x_adv)['out'])
    return lc + alpha_1 * la + beta_1 * lr

# --- 6) Training + 2-way Validation ---
train_losses = []
val_losses_clean, val_losses_pert = [], []
val_accs_clean, val_accs_pert = [], []
lrs = []
alpha_1 = 1.0
beta_1 = 0.5
batch_size = 2


for ep in range(1, epochs + 1):
    # --- Adjust learning rate ---
    if ep <= warmup_epochs:
        current_lr = adjust_lr(optimizer, ep - 1, warmup_epochs, warmup_lr, base_lr)
    else:
        main_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)

    # --- Train ---
    model.train()
    running = 0.0
    for x_adv, masks in train_loader:
        x_adv, masks = x_adv.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = composite_loss(model, x_adv, masks, x_adv, masks, alpha_1=1.0, beta_1=0.5)
        loss.backward()
        optimizer.step()
        running += loss.item()
    
    train_loss = running / len(train_loader)
    train_losses.append(train_loss)

    # --- Val clean & perturbed ---
    model.eval()
    rc, rp = 0.0, 0.0  # Losses
    cc, cp = 0, 0      # Correct
    tc, tp = 0, 0      # Total

    with torch.no_grad():
        for (imgs_c, masks_c), (imgs_p, masks_p) in zip(val_clean_loader, val_pert_loader):
            # Clean
            imgs_c, masks_c = imgs_c.to(device), masks_c.to(device)
            out_c = model(imgs_c)['out']
            rc += F.cross_entropy(out_c, masks_c).item()
            preds_c = out_c.argmax(1)
            cc += (preds_c == masks_c).sum().item()
            tc += masks_c.numel()

            # Perturbed
            imgs_p, masks_p = imgs_p.to(device), masks_p.to(device)
            out_p = model(imgs_p)['out']
            rp += F.cross_entropy(out_p, masks_p).item()
            preds_p = out_p.argmax(1)
            cp += (preds_p == masks_p).sum().item()
            tp += masks_p.numel()

    val_clean_loss = rc / len(val_clean_loader)
    val_pert_loss = rp / len(val_pert_loader)
    val_clean_acc = cc / tc
    val_pert_acc = cp / tp

    val_losses_clean.append(val_clean_loss)
    val_losses_pert.append(val_pert_loss)
    val_accs_clean.append(val_clean_acc)
    val_accs_pert.append(val_pert_acc)

    # Log message với thời gian
    elapsed_time = time.time() - start_time
    print(f"Epoch {ep}/{epochs}"
          f" | LR: {current_lr:.6f}"
          f" | Train: {train_loss:.4f}"
          f" | Val Clean: {val_clean_loss:.4f}, Acc: {val_clean_acc:.4f}"
          f" | Val Pert: {val_pert_loss:.4f}, Acc: {val_pert_acc:.4f}"
          f" | Time elapsed: {elapsed_time/60:.2f} minutes")

    # Fallback scheduler dựa trên perturbed val loss
    if ep > warmup_epochs:
        fallback_scheduler.step(val_pert_loss)

# ---  Vẽ đồ thị ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses_clean, label="Val Clean Loss")
plt.plot(val_losses_pert, label="Val Perturbed Loss")
plt.legend()
plt.title("Loss Curves")
plt.savefig('/kaggle/working/loss_curves.png')
plt.close()

plt.figure()
plt.plot(val_accs_clean, label="Val Clean Acc")
plt.plot(val_accs_pert, label="Val Perturbed Acc")
plt.legend()
plt.title("Accuracy")
plt.savefig('/kaggle/working/accuracy.png')
plt.close()

plt.figure()
plt.plot(lrs, label="Learning Rate")
plt.legend()
plt.title("Learning Rate Schedule")
plt.savefig('/kaggle/working/lr_schedule.png')
plt.close()

# --- 8) Lưu checkpoint ---
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'main_scheduler_state_dict': main_scheduler.state_dict(),
    'train_losses': train_losses,
    'val_losses_clean': val_losses_clean,
    'val_losses_pert': val_losses_pert,
    'val_accs_clean': val_accs_clean,
    'val_accs_pert': val_accs_pert,
    'lrs': lrs,
    'best_chromo': best_chromo,
    'hyperparameters': {
        'epochs': epochs,
        'warmup_epochs': warmup_epochs,
        'base_lr': base_lr,
        'warmup_lr': warmup_lr,
        'alpha_1': 1.0,
        'beta_1': 0.5,
        'batch_size': 2
    }
}
torch.save(checkpoint, '/kaggle/working/checkpoint.pth')
print(f"Checkpoint lưu tại batch {batch_idx+1}")

# Lưu modelA (nếu cần)
torch.save(modelA.state_dict(), '/kaggle/working/deeplabv3_lyftudacity.pth')
torch.save(model.state_dict(), '/kaggle/working/new_model_on_perturbed.pth.pth')

# --- (Tùy chọn) Tải lại checkpoint ---
def load_checkpoint(checkpoint_path, model, optimizer, main_scheduler):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    main_scheduler.load_state_dict(checkpoint['main_scheduler_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses_clean = checkpoint['val_losses_clean']
    val_losses_pert = checkpoint['val_losses_pert']
    val_accs_clean = checkpoint['val_accs_clean']
    val_accs_pert = checkpoint['val_accs_pert']
    lrs = checkpoint['lrs']
    best_chromo = checkpoint['best_chromo']
    hyperparameters = checkpoint['hyperparameters']
    return (train_losses, val_losses_clean, val_losses_pert, val_accs_clean, val_accs_pert, lrs, best_chromo, hyperparameters)

# Ví dụ tải lại (bỏ comment nếu cần):
# train_losses, val_losses_clean, val_losses_pert, val_accs_clean, val_accs_pert, lrs, best_chromo, hyperparameters = load_checkpoint('/kaggle/working/checkpoint.pth', model, optimizer, main_scheduler)