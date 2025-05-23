# --- 0) Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1) Tạo Dataset cho Lyft-Udacity Challenge ---
class LyftUdacityDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.label_paths = []

        folders = ['A', 'B', 'C', 'D', 'E']
        for i in folders:
            img_dir = os.path.join(root_dir, f"data{i}", f"data{i}", "CameraRGB")
            label_dir = os.path.join(root_dir, f"data{i}", f"data{i}", "CameraSeg")
            
            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                print(f"Thư mục không tồn tại: {img_dir} hoặc {label_dir}")
                continue
            
            img_files = sorted(os.listdir(img_dir))
            label_files = sorted(os.listdir(label_dir))
            
            for img_file, label_file in zip(img_files, label_files):
                self.image_paths.append(os.path.join(img_dir, img_file))
                self.label_paths.append(os.path.join(label_dir, label_file))

        total_samples = len(self.image_paths)
        train_size = int(0.8 * total_samples)
        if split == 'train':
            self.image_paths = self.image_paths[:train_size]
            self.label_paths = self.label_paths[:train_size]
        else:
            self.image_paths = self.image_paths[train_size:]
            self.label_paths = self.label_paths[train_size:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])
        label = np.array(label)
        label = label[:, :, 0].astype(np.uint8)  # Chuyển sang uint8 thay vì int64

        # Resize cả image và label
        if self.transform:
            # Transform cho image
            image = self.transform(image)
            # Resize label để khớp với kích thước hình ảnh sau transform
            label = Image.fromarray(label)
            label = T.Resize((512, 800), interpolation=T.InterpolationMode.NEAREST)(label)
            label = np.array(label, dtype=np.int64)  # Chuyển lại sang int64 sau khi resize

        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Transform cho hình ảnh
transform = T.Compose([
    T.Resize((512, 800)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tạo dataset và DataLoader
root_dir = '/kaggle/input/lyft-udacity-challenge'
dataset = LyftUdacityDataset(root_dir, split='train', transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Hiển thị ngẫu nhiên 3 mẫu ---
number_of_samples = len(dataset.image_paths)

for i in range(3):
    N = random.randint(0, number_of_samples - 1)
    img = Image.open(dataset.image_paths[N]).convert('RGB')
    mask = Image.open(dataset.label_paths[N])
    mask = np.array(mask)
    mask = mask[:, :, 0]  # Lấy channel đỏ (R) thay vì max qua các channel

    # Denormalize hình ảnh để hiển thị (bỏ chuẩn hóa ImageNet)
    img_np = np.array(img) / 255.0  # Đưa về [0, 1] nếu chưa chuẩn hóa
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = (img_np.transpose(2, 0, 1) * std + mean).transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    fig, arr = plt.subplots(1, 3, figsize=(20, 8))
    arr[0].imshow(img_np)
    arr[0].set_title('Image')
    arr[0].axis("off")
    arr[1].imshow(mask, cmap='gray')
    arr[1].set_title('Segmentation (Gray)')
    arr[1].axis("off")    
    arr[2].imshow(mask, cmap='Paired')
    arr[2].set_title('Segmentation (Paired)')
    arr[2].axis("off")
    plt.show()