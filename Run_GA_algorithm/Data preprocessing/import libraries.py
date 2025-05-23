import torch as nn
import torch.optim as optim
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
import torchattacks
import numpy as np
import random
import time
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
from scipy.stats import mode
import torch.nn.functional as F
import os
import copy
import imageio
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"