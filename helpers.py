import os
import sys
import math
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

def set_random_seed(seed: int) -> None:
    """Fix random seeds to ensure exact reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def crop_to_even(img):
    """Ensure image dimensions are even, which is required for neural network inputs."""
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2] 
        new_h, new_w = h - (h % 2), w - (w % 2)
        if h != new_h or w != new_w:
            return img[:new_h, :new_w]
        return img
    elif hasattr(img, 'size') and isinstance(img.size, tuple): 
        w, h = img.size
        new_w, new_h = w - (w % 2), h - (h % 2)
        if w != new_w or h != new_h:
            return img.crop((0, 0, new_w, new_h))
        return img
    return img

def add_gaussian_noise(tensor, std_dev):
    """Add Gaussian noise based on the provided standard deviation."""
    if std_dev <= 0:
        return tensor
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise

def convert(img):
    """Scale and clip image tensor back to 8-bit [0-255] format."""
    result = img * 255
    result = result * (result > 0)
    result = result * (result <= 255) + 255 * (result > 255)
    result = result.astype(np.uint8)
    return result

def apply_f(x):
    """Apply forward Fourier transform and return the magnitude."""
    d = x.shape[2]
    if x.shape[1] == 3:
        # Process RGB channels separately
        (r, g, b) = torch.split(x, [1, 1, 1], dim=1)
        r_, g_, b_ = torch.fft.fftn(r), torch.fft.fftn(g), torch.fft.fftn(b)
        y = torch.cat((torch.abs(r_), torch.abs(g_), torch.abs(b_)), 1)
    else:
        y = torch.fft.fftn(x)
        y = torch.abs(y)
    return y

def fftn(x):
    """Perform N-dimensional Fast Fourier Transform."""
    d = x.shape[2]
    if x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim=1)
        r_, g_, b_ = torch.fft.fftn(r), torch.fft.fftn(g), torch.fft.fftn(b)
        y = torch.cat((r_, g_, b_), 1)
    else:
        y = torch.fft.fftn(x)
    return y

def ifftn(x):
    """Perform N-dimensional Inverse Fast Fourier Transform and return real parts."""
    if x.shape[1] == 1:
        y = torch.fft.ifftn(x).real
    elif x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim=1)
        r_, g_, b_ = torch.fft.ifftn(r).real, torch.fft.ifftn(g).real, torch.fft.ifftn(b).real
        y = torch.cat((r_, g_, b_), 1)
    return y

def tv_grad2(x):
    """Compute Total Variation (TV) gradient."""
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    weight1 = torch.Tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]])
    weight2 = torch.Tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]])

    p = torch.zeros([1, c, h, w], dtype=torch.float)
    q = torch.zeros([1, c, h, w], dtype=torch.float)
    z = torch.zeros([1, c, h, w], dtype=torch.float)
    p[:, :, 0:h-1, :] = x[:, :, :-1, :] - x[:, :, 1:, :]
    q[:, :, :, 0:w-1] = x[:, :, :, :-1] - x[:, :, :, 1:]
    z = (p**2 + q**2)**(1./2)
    p = torch.from_numpy(np.nan_to_num((p / z).numpy()))
    q = torch.from_numpy(np.nan_to_num((q / z).numpy()))

    if c == 1:
        res1 = nn.functional.conv2d(p, weight1, padding=1, bias=None)
        res2 = nn.functional.conv2d(q, weight2, padding=1, bias=None)
    elif c == 3:
        res1_0 = nn.functional.conv2d(p[:, 0:1, :, :], weight1, padding=1, bias=None)
        res1_1 = nn.functional.conv2d(p[:, 1:2, :, :], weight1, padding=1, bias=None)
        res1_2 = nn.functional.conv2d(p[:, 2:3, :, :], weight1, padding=1, bias=None)
        res1 = torch.cat((res1_0, res1_1, res1_2), 1)
        res2_0 = nn.functional.conv2d(q[:, 0:1, :, :], weight2, padding=1, bias=None)
        res2_1 = nn.functional.conv2d(q[:, 1:2, :, :], weight2, padding=1, bias=None)
        res2_2 = nn.functional.conv2d(q[:, 2:3, :, :], weight2, padding=1, bias=None)
        res2 = torch.cat((res2_0, res2_1, res2_2), 1)
    return res1 + res2

def np_to_tensor(img_np):
    """Convert numpy array (C x W x H [0..1]) to torch.Tensor."""
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    """Convert numpy array to torch.Variable with batch dimension (1 x C x W x H)."""
    return Variable(np_to_tensor(img_np)[None, :])

def pil_to_np(img_PIL):
    """Convert PIL image (W x H x C [0...255]) to numpy array (C x W x H [0..1])."""
    ar = np.array(img_PIL)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255. 

def mse(x_hat, x_true):
    """Calculate normalized Mean Squared Error."""
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse_val = np.mean(np.square(x_hat/1.0 - x_true/1.0))
    energy = np.mean(np.square(x_true))   
    return mse_val / energy

def rgb2gray(rgb):
    """Convert RGB tensor to grayscale numpy array."""
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array([gray])
    
def num_param(net):
    """Count the total number of parameters in the network."""
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    return s

def gamma_correction(image, gamma):
    """Apply gamma correction to the image."""
    return image ** gamma