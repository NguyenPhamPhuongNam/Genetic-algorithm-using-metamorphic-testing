class SegmentationModel:
    def __init__(self, modelA, device):
        self.model = modelA
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def predict(self, img_np):
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)['out']
            pred_mask = pred.argmax(1).squeeze().cpu().numpy()
        return pred_mask

seg_model = SegmentationModel(modelA, device)

