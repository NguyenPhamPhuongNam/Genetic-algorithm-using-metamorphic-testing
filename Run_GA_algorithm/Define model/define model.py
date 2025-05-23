# --- 2) Model DeepLabV3 ---
modelA = deeplabv3_resnet50(weights="DEFAULT").to(device)
modelA.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=(1, 1), stride=(1, 1)) 



model = fcn_resnet50(pretrained=False, num_classes=13).to(device) # vi lyft_udacity_challenge co 13 lop tong cong 