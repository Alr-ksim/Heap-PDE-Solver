import argparse

import math
import json
import time
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


import numpy as np
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
from network import V, U_HJB, U_PRICE, U_TWO, Adaptive_PDQP_Net, PDQP_Net_2_full
from calculation import d_ball_uniform, laplacian, du_dtheta_
from equation import EqnConfig, PricingDefaultRisk, PricingDiffRate, BurgersType, ReactionDiffusion, HJBLQ, AllenCahn, HeatEquation

# Generate test dataset
def fit_theta_T(g_param, u, eqn, n_x=500, theta_init=None, n_iter=2000, seed=1, thereshold=1e-3):
    '''
    Inputs:
        g_param: (bs, 1)
        u: instance of network U
        eqn: instance of PricingDefaultRisk
        n_x: number of samples points used in outer optimizer.
        theta_init:  tensor, initial value of theta
        n_iter: number of iterations. 2000 for 10dim, 3000 for 15dim
    Outputs:
        theta: (bs, n_params)
    '''
    
    bs = g_param.shape[0]
    device = g_param.device
    if not theta_init is None:
        theta = nn.Parameter(theta_init.clone())   #Note: Get the initial theta if given
    else:
        # fixed initialization.
        generator = torch.Generator(device=device).manual_seed(seed)
        theta = nn.Parameter(torch.randn((bs, u.n_params), device=device, dtype=g_param.dtype, generator=generator))
    
    # minimize ||f(x) - u(x)||^2 by Adam
    optimizer = torch.optim.Adam([theta], lr=0.001)
    scaler = torch.cuda.amp.GradScaler()  #NOTE: use mixed precision training, save memory.


    for i in range(0,n_iter+1): 
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if i % 50 == 0: #update x every 50 iterations, save computation.
                _, x = eqn.sample(bs * n_x, device)  # x.shape = (bs*n_x, dim, eqn.num_time_interval+1)
                x = x[:, :, -1]  # shape=(bs*n_x, dim)
                x = x.reshape(bs, n_x, eqn.dim)  # shape=(bs, n_x, dim)

                # Temp t set
                t = torch.full((bs, n_x, 1), eqn.total_time, device=device)
                true_sol = eqn.g(t=t, x=x, g_param=g_param)  # (bs, n_x)
    
            pred_sol = u.forward_2(theta, x)
            relative_error = ((true_sol - pred_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)
            loss = relative_error.mean()
            if loss < thereshold:
                break
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return theta.detach(), relative_error.detach()   # (bs, n_params), (bs, )

def gen_rand_test_dataset(u, eqn_config, n_samples=500, device=None):
    theta_0 = torch.randn(n_samples, u.n_params)
    test_dataset = torch.utils.data.TensorDataset(theta_0)
    filename = f"checkpoints/{eqn_config.eqn_type}_dataset_dim{eqn_config.dim}_params{u.n_params}_seed{eqn_config.seed}_rand.pt"
    torch.save(test_dataset, filename)
    print(f"dataset generated, len={len(test_dataset)} saved to ", filename)
    return test_dataset
    
def gen_test_dataset(u, eqn, eqn_config, device):
    '''
    Output a tensor dataset with at least n data samples, with 4 tensors:
        theta_T: tensor (n, n_params)
        x: tensor (n, n_x, dim)
        p: tensor (n, n_x, )
        u_0: tensor (n, n_x,),  
    '''
    with open(f"checkpoints/{eqn_config.eqn_type}_d{eqn_config.dim}_combined_records.json", 'r') as f:
        data = json.load(f)

    n = len(data)
    test_dataset = [[], [], [], []]
    for i in range(n):
        g_param = torch.tensor([data[i]['g_param']]).float().to(device) #shape=(1,)
        g_param = g_param.unsqueeze(0) #shape=(1, 1)
        x = []
        u_0 = []
        for d in data[i]['data']:
            if d[0] == eqn.total_time:  #only save the t = T data
                x.append(d[1])
                u_0.append(d[2])
        x = torch.tensor(x).float().to(device).unsqueeze(0) #shape=(1, n_x, dim) 
        u_0 = torch.tensor(u_0).float().to(device).T #shape=(1, n_x,)
        p = torch.ones_like(u_0) #shape=(1, n_x)

        test_dataset[0].append(g_param)
        test_dataset[1].append(x)
        test_dataset[2].append(p)
        test_dataset[3].append(u_0)

    test_dataset = [torch.cat(test_dataset[i], dim=0) for i in range(4)]

    seed = eqn_config.seed
    bs = eqn_config.fit_bs
    n_iter = eqn_config.fit_n_iter
    thereshold = eqn_config.fit_thereshold

    generator = torch.Generator(device=device).manual_seed(seed)
    theta_init_ = torch.randn((bs, u.n_params), device=device, dtype=g_param.dtype, generator=generator)
    theta_T_list = []
    l2re_list = []
    for i in range(n // bs):
        g_param = test_dataset[0][i*bs:(i+1)*bs]
        theta_T, l2re = fit_theta_T(g_param, u, eqn, seed=seed, theta_init=theta_init_, n_iter=n_iter, thereshold=thereshold) #shape = (bs, n_params)
        print(f"l2re={l2re.mean().item()}")
        theta_T_list.append(theta_T)
        l2re_list.append(l2re)
    theta_T = torch.cat(theta_T_list, dim=0)
    l2re = torch.cat(l2re_list, dim=0)
    test_dataset = [theta_T, test_dataset[1], test_dataset[2], test_dataset[3]]
    n = len(theta_T)
    test_dataset = [i[:n][l2re < 1e-3] for i in test_dataset] # Key, origin is 1.1e-3

    test_dataset = torch.utils.data.TensorDataset(*test_dataset)
    filename = f"checkpoints/{eqn_config.eqn_type}_dataset_dim{eqn_config.dim}_params{u.n_params}_seed{eqn_config.seed}.pt"
    torch.save(test_dataset, filename)
    print(f"dataset generated, len={len(test_dataset)} saved to ", filename)
    return test_dataset

def test_gen_test_dataset(eqn_config, with_label):
    if eqn_type == "pricing_default_risk":
        eqn = PricingDefaultRisk(eqn_config)
    elif eqn_type == "pricing_diffrate":
        eqn = PricingDiffRate(eqn_config)
    elif eqn_type == "burgers_type":
        eqn = BurgersType(eqn_config)
    elif eqn_type == "reaction_diffusion":
        eqn = ReactionDiffusion(eqn_config)
    elif eqn_type == "hjb_lq":
        eqn = HJBLQ(eqn_config)
    elif eqn_type == "allencahn":
        eqn = AllenCahn(eqn_config)
    elif eqn_type == 'heat':
        eqn = HeatEquation(eqn_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = U_TWO(eqn_config.dim, hidden_dim1=40, hidden_dim2=40).to(device)
    # u = U_PRICE(eqn_config.dim, width=200).to(device)    # For pricing diffrate equation
    
    if with_label:
        gen_test_dataset(u=u, eqn=eqn, eqn_config=eqn_config, device=device)
    else:
        n_test_samples = 250
        gen_rand_test_dataset(u=u, eqn_config=eqn_config, n_samples=n_test_samples, device=device)


# Generate train dataset
def gen_train_dataset(n_params, n_samples):
    '''training dataset, only initial values.'''
    n_1 = n_samples // 5
    n_2 = n_samples - n_1
    theta_0_1 = d_ball_uniform(n_1, n_params, n_params/100)  # n_params=2000, n_params/100=20, i.e. |theta_0_1| < 20 
    theta_0_2 = torch.randn(n_2, n_params)*math.sqrt(0.5)
    theta_0 = torch.cat((theta_0_1, theta_0_2), dim=0)
    
    dataset = torch.utils.data.TensorDataset(theta_0)  #（n_1+n_2, n_params）
    return dataset


# Train and evaluation
def loss_(theta_0, u, v, pdqp, eqn, pdqp_flag, c_type=0, verbose=False):
    n_x = 1000  #NOTE: original:10000
    bs  = theta_0.shape[0]
    T = eqn.total_time
    n_t = eqn.num_time_interval
    device = theta_0.device
    x_all = eqn.sample(bs*n_x, device=device)[1] #shape=(bs*n_x, dim, n_t+1)
    x_all = x_all.reshape(bs, n_x, eqn.dim, n_t+1)
    p_x = 1.0

    # --- constants of PDQP-Net ---
    n_param = u.n_params
    
    # Constraints: X <= u, X >= l
    pdqp_l = -100 * torch.ones((bs, n_param), device=device)  #NOTE: l and u must be large enough during training.
    pdqp_u = 100 * torch.ones((bs, n_param), device=device)
    
    # initial solutions
    pdqp_x_init = torch.zeros((bs, n_param), device=device)
    pdqp_y_init = torch.zeros((bs, n_x), device=device)

    # regularizer
    pdqp_reg = 1e-5 * torch.eye(n_param, device=device).unsqueeze(0).repeat(bs, 1, 1)
    
    loss_c_container = [None]

    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of three torch.Tensor, theta, r and r_c, shape (bs, n_param) (1, ) and (1,)
        '''
        x = x_all[..., - 1 -int(n_t * t/T)] #shape=(bs, n_x, dim),  time dependent sampling, time backward.
        theta, r = gamma
        dtheta_dt_init_guess = v(theta) ##shape=(bs, n_param)

        #--- calculate dr_dt (Eq.5) via Importance Sampling---
        theta_rep = theta.unsqueeze(1).repeat(1, n_x,1)  #shape=(bs, n_x, n_param)
        du_dtheta = du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        t_g = torch.full((x.shape[0], x.shape[1]), T, device=device)
        rhs = eqn.rhs(u, theta_rep, x, t_g)  #shape=(bs, n_x)
        if c_type == 0:
            pdqp_b = -1 * eqn.lamb * torch.ones((bs, n_x), device=device)
        elif c_type == 1:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach()
        elif c_type == 2:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach() - eqn.lamb * laplacian(u, theta, x).detach()
        else:
            pdqp_b = -100 * torch.ones((bs, n_x), device=device)

        if (pdqp_flag):
            # --- PDQP-Net Input ---
            P = du_dtheta  # shape=(bs, n_x, n_param)
            Q = torch.matmul(P.transpose(-1, -2), P)  # Q = P^T P, shape=(bs, n_param, n_param)
            Q = Q + pdqp_reg  # NOTE: Add a small positive diagonal term to Q
            c = - torch.matmul(P.transpose(-1, -2), rhs.unsqueeze(-1))  # c = - P^T rhs
            c = c.squeeze(-1) # shape=(bs, n_param) 
            
            # Constraints: du/dt >= r*u (for pricing default risk)
            if c_type == 1 or c_type == 2:
                A = P
            else:
                A = -1.0 * P
            # --- PDQP-Net Output ---
            x_hat, y_hat = pdqp(dtheta_dt_init_guess, pdqp_y_init, Q, A, c, pdqp_b, pdqp_l, pdqp_u, theta)
            dtheta_dt = x_hat
        else:
            dtheta_dt = dtheta_dt_init_guess

        du_dt = du_dtheta * dtheta_dt.unsqueeze(1)  #shape=(bs, n_x, n_param)
        du_dt = du_dt.sum(-1)  #shape=(bs, n_x)
        lhs = du_dt
        
        dr_dt = ((lhs - rhs).pow(2) / p_x).mean()
        
        d_pc = (torch.norm(torch.relu(pdqp_b - du_dt), p=2, dim=(0, 1)) / torch.norm(du_dt, p=2, dim=(0, 1))).mean()
        
        d_ns = 1 - (torch.norm(torch.clamp(dtheta_dt, pdqp_l, pdqp_u), p=2, dim=(1)) / torch.norm(dtheta_dt, p=2, dim=(1))).mean()
        
        r1 = 0.01
        r2 = 1.0
        r3 = 10.0
        
        loss = r1 * dr_dt + r2 * d_pc + r3 * d_ns
        loss_c = d_pc + d_ns

        if t >= (n_t*4-1)*T/(n_t*4):
            loss_c_container[0] = loss_c.detach()
            if verbose:
                print(f"Loss_c in the last step: [{loss_c.item():.4f}]")
        return (dtheta_dt, loss)
    
    r_0 = torch.zeros(1).to(device) # initial condition for loss
    r_c = torch.zeros(1).to(device) # initial condition for constraint loss
    t = torch.tensor([0, T]).to(device)  # time grid
    traj = odeint_adjoint(ode_func, y0=(theta_0, r_0), t=t, method='rk4', options={'step_size':T/n_t}, 
                          adjoint_params=list(v.parameters()) + list(pdqp.parameters()))
    theta_traj, r_traj = traj

    r_T = r_traj[-1] # final condition for r
    loss_c_final = loss_c_container[0]
    return r_T, loss_c_final # shape = (1,)

def inference(theta_0, u, v, pdqp, eqn, pdqp_flag, c_type=None):
    n_x = 1000  #NOTE: original:10000
    bs  = theta_0.shape[0]
    T = eqn.total_time
    n_t = eqn.num_time_interval
    device = theta_0.device
    x_all = eqn.sample(bs*n_x, device=device)[1] #shape=(bs*n_x, dim, n_t+1)
    x_all = x_all.reshape(bs, n_x, eqn.dim, n_t+1)
    p_x = 1.0

    # --- constants of PDQP-Net ---
    n_param = u.n_params
    
    # Constraints: X <= u, X >= l
    pdqp_l = -100 * torch.ones((bs, n_param), device=device)  #NOTE: l and u must be large enough during training.
    pdqp_u = 100 * torch.ones((bs, n_param), device=device)
    
    # initial solutions
    pdqp_x_init = torch.zeros((bs, n_param), device=device)
    pdqp_y_init = torch.zeros((bs, n_x), device=device)

    # regularizer
    pdqp_reg = 1e-5 * torch.eye(n_param, device=device).unsqueeze(0).repeat(bs, 1, 1)

    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of three torch.Tensor, theta, r and r_c, shape (bs, n_param) (1, ) and (1,)
        '''
        x = x_all[..., - 1 -int(n_t * t/T)] #shape=(bs, n_x, dim),  time dependent sampling, time backward.
        theta = gamma
        dtheta_dt_init_guess = v(theta) ##shape=(bs, n_param)

        #--- calculate dr_dt (Eq.5) via Importance Sampling---
        theta_rep = theta.unsqueeze(1).repeat(1, n_x,1)  #shape=(bs, n_x, n_param)
        du_dtheta = du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        t_g = torch.full((x.shape[0], x.shape[1]), T, device=device)
        rhs = eqn.rhs(u, theta_rep, x, t_g)  #shape=(bs, n_x)
        if c_type == 0:
            pdqp_b = -1 * eqn.lamb * torch.ones((bs, n_x), device=device)
        elif c_type == 1:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach()
        elif c_type == 2:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach() - eqn.lamb * laplacian(u, theta, x).detach()
        else:
            pdqp_b = -100 * torch.ones((bs, n_x), device=device)

        if (pdqp_flag):
            # --- PDQP-Net Input ---
            P = du_dtheta  # shape=(bs, n_x, n_param)
            Q = torch.matmul(P.transpose(-1, -2), P)  # Q = P^T P, shape=(bs, n_param, n_param)
            Q = Q + pdqp_reg  # NOTE: Add a small positive diagonal term to Q
            c = - torch.matmul(P.transpose(-1, -2), rhs.unsqueeze(-1))  # c = - P^T rhs
            c = c.squeeze(-1) # shape=(bs, n_param) 
            
            # Constraints: du/dt >= r*u (for pricing default risk)
            if c_type == 1 or c_type == 2:
                A = P
            else:
                A = -1.0 * P
            # --- PDQP-Net Output ---
            x_hat, y_hat = pdqp(dtheta_dt_init_guess, pdqp_y_init, Q, A, c, pdqp_b, pdqp_l, pdqp_u, theta)
            dtheta_dt = x_hat
        else:
            dtheta_dt = dtheta_dt_init_guess

        return dtheta_dt
    
    r_0 = torch.zeros(1).to(device) # initial condition for loss
    t = torch.tensor([0, T]).to(device)  # time grid
    traj = odeint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':T/n_t})
    thetaT_pred = traj[-1]
    return thetaT_pred

def inference_2(theta_0, u, v, pdqp, x, eqn, pdqp_flag, c_type=None):
    n_x = x.shape[1]
    bs  = theta_0.shape[0]
    T = eqn.total_time
    n_t = eqn.num_time_interval
    device = theta_0.device
    p_x = 1

    # --- constants of PDQP-Net ---
    n_param = u.n_params
    
    # Constraints: X <= u, X >= l
    pdqp_l = -100 * torch.ones((bs, n_param), device=device)  #NOTE: l and u must be large to cover the range of theta during training.
    pdqp_u = 100 * torch.ones((bs, n_param), device=device)
    
    # initial solutions
    pdqp_x_init = torch.zeros((bs, n_param), device=device)
    pdqp_y_init = torch.zeros((bs, n_x), device=device)

    # regularizer
    pdqp_reg = 1e-5 * torch.eye(n_param, device=device).unsqueeze(0).repeat(bs, 1, 1)

    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of three torch.Tensor, theta, r and r_c, shape (bs, n_param) (1, ) and (1,)
        '''
        theta, res = gamma
        dtheta_dt_init_guess = v(theta) ##shape=(bs, n_param)

        #--- calculate dr_dt (Eq.5) via Importance Sampling---
        theta_rep = theta.unsqueeze(1).repeat(1, n_x,1)  #shape=(bs, n_x, n_param)
        du_dtheta = du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        t_g = torch.full((x.shape[0], x.shape[1]), T, device=device)
        rhs = eqn.rhs(u, theta_rep, x, t_g)  #shape=(bs, n_x)
        if c_type == 0:
            pdqp_b = -1 * eqn.lamb * torch.ones((bs, n_x), device=device)
        elif c_type == 1:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach()
        elif c_type == 2:
            pdqp_b = eqn.rate * u.forward_2(theta, x).detach() - eqn.lamb * laplacian(u, theta, x).detach()
        else:
            pdqp_b = -100 * torch.ones((bs, n_x), device=device)

        if (pdqp_flag):
            # --- PDQP-Net Input ---
            P = du_dtheta  # shape=(bs, n_x, n_param)
            Q = torch.matmul(P.transpose(-1, -2), P)  # Q = P^T P, shape=(bs, n_param, n_param)
            Q = Q + pdqp_reg  # NOTE: Add a small positive diagonal term to Q
            c = - torch.matmul(P.transpose(-1, -2), rhs.unsqueeze(-1))  # c = - P^T rhs
            c = c.squeeze(-1) # shape=(bs, n_param) 
            
            # Constraints: du/dt >= r*u (for pricing default risk)
            if c_type == 1 or c_type == 2:
                A = P
            else:
                A = -1.0 * P
            # --- PDQP-Net Output ---
            x_hat, y_hat = pdqp(dtheta_dt_init_guess, pdqp_y_init, Q, A, c, pdqp_b, pdqp_l, pdqp_u, theta)
            dtheta_dt = x_hat
        else:
            dtheta_dt = dtheta_dt_init_guess

        d_res = torch.zeros(res.shape).to(device)
        
        # if t >= (n_t*4-1)*T/(n_t*4):
        if True:
            du_dt = du_dtheta * dtheta_dt.unsqueeze(1)  #shape=(bs, n_x, n_param)
            du_dt = du_dt.sum(-1)  #shape=(bs, n_x)
            lhs = du_dt
            d_res = ((lhs - rhs).pow(2) / p_x)
            
        return dtheta_dt, d_res
    
    
    res = torch.zeros((bs, n_x), device=device)
    t = torch.tensor([0, T]).to(device)  #time grid
    traj = odeint(ode_func, y0=(theta_0, res), t=t, method='rk4', options={'step_size':T/n_t})
    theta_traj, r_traj = traj
    thetaT_pred = theta_traj[-1]
    res = r_traj[-1] / (4*n_t)
    return thetaT_pred, res

def test(u, data, pred_theta_0, normed=True):
    '''function distance between label and samples of true sol.'''
    device = pred_theta_0.device
    x = data[1].to(device)
    p_x = data[2].to(device)
    label_u_0 = data[3].to(device)
    pred_u_0 = u.forward_2(pred_theta_0, x)
    if normed:
        err = (((label_u_0 - pred_u_0)**2 / p_x).mean(dim=1) / (label_u_0**2 / p_x).mean(dim=1)).mean()
    else:
        err = ((label_u_0 - pred_u_0)**2 / p_x).mean() 
    return err

def eval(u, v, pdqp, pdqp_flag, eqn, dataloader, with_label, device, c_type=0, verbose=False):
    '''eval v based on pre-generated test_dataset.'''
    v.eval()
    pdqp.eval()

    with torch.no_grad():
        loss_list = []
        c_loss_list = []
        for data in dataloader:
            theta_T = data[0].to(device).float()
            if with_label:
                pred_theta_0 = inference(theta_T, u, v, pdqp, eqn, pdqp_flag)
                L2RE = test(u, data, pred_theta_0)
                loss_list.append(L2RE)
                c_loss_list.append(0.0*L2RE)    # No constraint loss for no constraint pde
            else:
                PINN_loss, C_loss = loss_(theta_T, u, v, pdqp, eqn, pdqp_flag, c_type=c_type, verbose=False)
                loss_list.append(PINN_loss)
                c_loss_list.append(C_loss)

        if verbose: # print last batch
            print(f"Loss_list = {torch.tensor(loss_list)}, C_loss_list = {torch.tensor(c_loss_list)}")
        return sum(loss_list)/len(loss_list), sum(c_loss_list)/len(c_loss_list)

def train(eqn_config, model_type='cso', drop_out=1, with_label=1, test_num=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = U_TWO(eqn_config.dim, hidden_dim1=40, hidden_dim2=40).to(device)  # target net
    # u = U_PRICE(eqn_config.dim, width=200).to(device)    # For pricing diffrate equation
    if eqn_type == "pricing_default_risk":
        eqn = PricingDefaultRisk(eqn_config) #test on the full T.
    elif eqn_type == "pricing_diffrate":
        eqn = PricingDiffRate(eqn_config)
    elif eqn_type == "burgers_type":
        eqn = BurgersType(eqn_config)
    elif eqn_type == "reaction_diffusion":
        eqn = ReactionDiffusion(eqn_config)
    elif eqn_type == "hjb_lq":
        eqn = HJBLQ(eqn_config)
    elif eqn_type == "allencahn":
        eqn = AllenCahn(eqn_config)
    elif eqn_type == 'heat':
        eqn = HeatEquation(eqn_config)
    
    if drop_out:
        dropout_p = eqn_config.train_dropout
    else:
        dropout_p = 0.0
    n_hiddens = eqn_config.train_hiddens
    pdqp_flag = False
    if model_type == 'heap':
        pdqp_flag  = True

    v = V(dim=u.n_params, width=n_hiddens, dropout_p=dropout_p).to(device) #node model
    v = v.to(device)

    # pdqp net
    c_type = eqn_config.c_type
    feat_sizes = [eqn_config.feat_size] * eqn_config.feat_depth
    
    pdqp = Adaptive_PDQP_Net(feat_sizes, dropout_p)
    pdqp = pdqp.to(device)

    # load datasets
    train_bs = eqn_config.train_bs
    test_bs = 250

    if with_label:
        dataset_test = torch.load(f'checkpoints/{eqn_config.eqn_type}_dataset_dim{eqn_config.dim}_params{u.n_params}_seed{eqn_config.seed}.pt', weights_only = False, map_location=device)
    else:
        dataset_test = torch.load(f'checkpoints/{eqn_config.eqn_type}_dataset_dim{eqn_config.dim}_params{u.n_params}_seed{eqn_config.seed}_rand.pt', weights_only = False, map_location=device)
    dataset_train = gen_train_dataset(n_params=u.n_params, n_samples=10000)

    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=train_bs, shuffle=True, drop_last=False,)
    dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_bs, shuffle=False, drop_last=False,)
    
    # train
    n_epoch = eqn_config.train_epoch
    lr = eqn_config.train_lr
    print(f"lr={lr}")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(v.parameters()) + list(pdqp.parameters())), lr=lr)  #original lr=5e-4

    start_time = time.time()
    loss_mark = None
    best_score = 1.0
    best_ckp = {'v_model_state_dict':None, 'pdqp_model_state_dict':None, 'optimizer_state_dict':None, 'test_l2re':1e10}
    os.makedirs("checkpoints", exist_ok=True)
    print(f"{v.__class__.__name__} {dropout_p=}, {eqn.total_time=}")
    print("num of trainable v params=", sum(p.numel() for p in v.parameters()))
    print("num of trainable pdqp params=", sum(p.numel() for p in pdqp.parameters()))
    
    for i in range(n_epoch):
        v.train()
        for j, data in enumerate(dataload_train):
            
            theta_0 = data[0].to(device) #shape=(bs, n_param)
            loss, c_loss = loss_(theta_0, u, v, pdqp, eqn, pdqp_flag, c_type=c_type, verbose=False)
            
            # No more skip first iter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if True:
                print('----------------------')
                print(f'epoch {i}, batch {j}, loss {loss.item()}')
                test_l2re, c_l2re = eval(u, v, pdqp, pdqp_flag, eqn, dataload_test, with_label, device, c_type=c_type)

                if loss_mark == None:
                    loss_mark = test_l2re.item()
                rel_loss = test_l2re.item()/loss_mark
                cur_score = 0.3*c_l2re.item() + 0.7*rel_loss
                print(f"Total loss in testset = [{test_l2re.item():.4f}]")
                print(f"Relative loss in testset = [{rel_loss:.4f}]")
                print(f"Constraint loss in testset = [{c_l2re.item():.4f}]")
                print(f'walltime = {time.time()-start_time}', flush=True)

                # if test_l2re < best_ckp['test_l2re']:
                if cur_score < best_score:
                    best_score = cur_score
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['v_model_state_dict'] = v.state_dict()
                    best_ckp['pdqp_model_state_dict'] = pdqp.state_dict()
                    best_ckp['optimizer_state_dict'] = optimizer.state_dict()
                    print(f"best ckp updated")
                    if test_num:
                        model_path = f"checkpoints/{eqn_config.eqn_type}_dim{eqn_config.dim}_drop{drop_out}_{model_type}_c{c_type}_n{test_num}.pth"
                    else:
                        model_path = f"checkpoints/{eqn_config.eqn_type}_dim{eqn_config.dim}_drop{drop_out}_{model_type}_c{c_type}.pth"
                    torch.save(best_ckp, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heap PDE solver")
    parser.add_argument('--eqn_type', type=str, default='burgers_type', required=True, help='Equation type')
    parser.add_argument('--cur_dim', type=int, default=10, required=True, help='Equation dim')
    parser.add_argument('--model_type', type=str, default='cso', required=True, help='Models selection: cso or heap')
    parser.add_argument('--drop_out', type=int, default=1, required=True, help='Dropout or not')
    parser.add_argument('--with_label', type=int, default=1, required=True, help='With PDE solution labels or not')
    parser.add_argument('--train_mode', type=int, default=1, required=True, help='Training mode: train(1) or data generation(0)')
    parser.add_argument('--test_num', type=int, default=0, help='test branch')

    args = parser.parse_args()
    eqn_type = args.eqn_type
    cur_dim = args.cur_dim
    model_type = args.model_type
    drop_out = args.drop_out
    with_label = args.with_label
    train_mode = args.train_mode
    test_num = args.test_num
    eqn_config = EqnConfig(eqn_type=eqn_type, dim=cur_dim, num=test_num)

    set_seed(eqn_config.seed)
    redirect_log_file(exp_name=f"{eqn_type}_d{cur_dim}_{model_type}_drop{drop_out}")
    set_gpu_max_mem(default_device=0, force=False)
    if not train_mode:
        test_gen_test_dataset(eqn_config, with_label)
    else:
        train(eqn_config, model_type, drop_out, with_label, test_num)