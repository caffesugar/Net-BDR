import os
import math
import copy
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

# Import from local helpers
from helpers import apply_f, fftn, ifftn

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, factor=0.5):
    """Decay learning rate by a specified factor periodically."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        print(f'\nLR is set to {lr}\n') 
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr
    return optimizer

def fit(net,
        num_channels, 
        img_clean_var,
        out_channels=1,
        k=0.2, 
        d=256, 
        net_input=None,
        decodetype='upsample',
        code='uniform', 
        opt_input=False, 
        find_best=False,
        OPTIMIZER='adam',
        LR=0.01, 
        numit_inner=20,
        print_inner=False,
        optim='bdr',
        OPTIMIZER2='None',
        LR_LS=0.02,
        num_iter=5000,
        lr_decay_epoch=0, 
        w=0.0,
        rho=1.0,
        ksi=0.0001,
        img_origin=None,
        weight_decay=0,
        X1=None, X2=None, X3=None, X4=None,
        is_net=False,
        gamma=1.0,
        device='cuda',
        dtype=torch.cuda.FloatTensor
       ):
    """
    Main optimization loop for phase retrieval.
    Implements the Network-embedded BDR (Net-BDR) and Direct Optimization methods.
    """
    m = d + 2 * round(k * d)
    
    if net_input is not None:
        net_input = net_input
    else:
        # Total upsample scale: 2^(num_scales-1)
        totalupsample = 2 ** (len(num_channels) - 1)
        width = int(d / totalupsample)
        height = int(d / totalupsample)
        shape = [1, num_channels[0], width, height]
        
        # Initialize latent code Z with uniform distribution
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1. / 10
        
    net_input_saved = net_input.data.detach()
    p = [t for t in net.decoder.parameters()]
    
    if opt_input:
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_truth = np.zeros(num_iter)
    residuals = np.zeros(num_iter)
    best_x = Variable(torch.zeros([1, out_channels, d, d]))
    
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(p, lr=LR, weight_decay=weight_decay)
        
    mse = torch.nn.MSELoss()
    
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0
        best_t = 0

    if optim == 'bdr':
        b = img_clean_var
        v = Variable(torch.zeros([1, out_channels, m, m]))
        
        # Initialize v_0 via inverse FFT
        v = ifftn(b).to(device)
        v.data[:, :, :, :round(k * d)] = torch.from_numpy(X1[:, :, :]).float()
        v.data[:, :, :, -round(k * d):] = torch.from_numpy(X2[:, :, :]).float()
        v.data[:, :, :round(k * d), round(k * d):-round(k * d)] = torch.from_numpy(X3[:, :, :]).float()
        v.data[:, :, -round(k * d):, round(k * d):-round(k * d)] = torch.from_numpy(X4[:, :, :]).float()
        xx = v
        
        # Main Iteration Loop
        for i in range(num_iter):
            if lr_decay_epoch != 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
                
            # Decay coefficient for network embedding
            mu = math.exp(-(max(0, i - 30) / 50) ** 2)
            v_prev = v.clone().detach()
            
            # Forward FFT and modulus projection
            Y1 = fftn(v)
            Y1 = b * Y1 / (torch.abs(Y1) + 1e-8)  # Added epsilon for numerical stability
            
            # Inverse FFT and relaxation
            x2 = ifftn(Y1).real
            x2 = gamma * x2 + (1 - gamma) * v
           
            # Inner loop: Optimize the untrained network
            if is_net and mu > 0.05:
                for j in range(numit_inner):
                    optimizer.zero_grad()
                    out = net(net_input.type(dtype))
                    loss_inner = mse(out, x2[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d])
                    loss_inner.backward()
                    optimizer.step()
                g = net(net_input.type(dtype))
            else:
                mu = 0
                g = 0      
                       
            # BDR Background Update
            inter = v - x2 + xx 
            v = inter
            
            # Combine network output with analytical update
            u = x2[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d]
            v.data[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d] = mu * g + (1 - mu) * u

            if i % 50 == 0:
                print(f'Iteration {i}')
                
            loss_updated = mse(v.data[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d], 
                               v_prev.data[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d])
            try:
                diff_norm = torch.norm(v - v_prev).item()
                prev_norm = torch.norm(v_prev).item()
                if prev_norm > 1e-9:
                    residuals[i] = diff_norm / prev_norm
                else:
                    residuals[i] = 0
            except:
                residuals[i] = 0
                
            # Track best performance based on convergence loss
            if find_best:
                if best_mse > 1.001 * loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    best_x = v[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d]
                    best_t = i
            else:
                best_x = v[:, :, round(k * d):round(k * d)+d, round(k * d):round(k * d)+d]
                best_t = i
                
        if find_best:
            net = best_net
            
    elif optim == 'direct':
        b = img_clean_var
        background = torch.zeros([1, out_channels, m, m]).to(device)
        pad_len = round(k * d)
        
        if X1 is not None:
            background.data[:, :, :, :pad_len] = torch.from_numpy(X1[:, :, :]).float()
            background.data[:, :, :, -pad_len:] = torch.from_numpy(X2[:, :, :]).float()
            background.data[:, :, :pad_len, pad_len:-pad_len] = torch.from_numpy(X3[:, :, :]).float()
            background.data[:, :, -pad_len:, pad_len:-pad_len] = torch.from_numpy(X4[:, :, :]).float()
        
        for i in range(num_iter):
            if lr_decay_epoch != 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
            
            optimizer.zero_grad()
            out_obj = net(net_input.type(dtype))
            out_padded = F.pad(out_obj, (pad_len, pad_len, pad_len, pad_len), "constant", 0)
            full_img = out_padded + background
            
            est_fft = fftn(full_img)
            est_mag = torch.abs(est_fft)
            
            loss = mse(est_mag, b)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Iteration {i:05d}   Loss (Eq4) {loss.item():.6f}')
            mse_wrt_truth[i] = loss.item()
            
            if find_best:
                if best_mse > loss.item():
                    best_mse = loss.item()
                    best_net = copy.deepcopy(net)
                    best_x = out_obj.detach() 
                    best_t = i
            else:
                best_x = out_obj.detach()
                best_t = i
                
    return mse_wrt_truth, net_input_saved, net, net_input, best_x, best_t, residuals