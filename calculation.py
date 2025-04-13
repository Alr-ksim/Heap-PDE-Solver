import torch
import numpy as np

def d_ball_uniform(num_samples, d, scale_max,**kwargs):
    scale_min = kwargs.get("scale_min",0)
    u = torch.randn((num_samples,d))
    norms = (torch.sum(u**2,1).unsqueeze(-1)) ** 0.5
    r = (scale_max-scale_min)*torch.rand((num_samples,1))+scale_min

    final_samples = r*u/norms
    return final_samples

def laplacian(u, theta_rep, x):
    '''div \cdot grad u.
    Inputs:
        u: instance of U.
        theta_rep: torch.Tensor, shape (bs, n_x, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
    Outputs:
        div_u: torch.Tensor, shape (bs, n_x)
    '''
    with torch.enable_grad():
        x.requires_grad_()
        du_dx = torch.autograd.grad(u.forward_2(theta_rep,x).sum(), x, create_graph=True)[0] # shape=x.shape= (bs, n_x, d)
        du_dx = du_dx.sum(dim=(0,1)) # shape=(d,)
    
        div_u = 0
        for i in range(x.shape[-1]):
            du_dxi_dxi =  torch.autograd.grad(du_dx[i], x, create_graph=True)[0][...,i].detach()
            div_u += du_dxi_dxi # shape=(bs, n_x)
    return div_u

def du_dtheta_(u, theta_rep, x):
    '''du/dtheta.
    Inputs:
        u: instance of U.
        theta_rep: torch.Tensor, shape (bs, n_x, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
    Outputs:
        du/dtheta: torch.Tensor, shape (bs, n_x, n_param)
    '''
    with torch.enable_grad():
        theta_rep.requires_grad_()
        u_output = u.forward_2(theta_rep,x).sum()        
        du_dtheta = torch.autograd.grad(u_output, theta_rep)[0].detach()  #shape=(bs, n_x, n_param)
        return du_dtheta