import torch
import math
import json

import numpy as np
import torch.nn.functional as F

class EqnConfig():
    def __init__(self, eqn_type, dim=None, num=None, total_time=None, num_time_interval=None):
        if num:
            config_file = f"configs/{eqn_type}_d{dim}_n{num}.json"
        else:
            config_file = f"configs/{eqn_type}_d{dim}.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.eqn_type = config['eqn_type']
        self.dim = config['dim']
        self.total_time = config['total_time']
        self.num_time_interval = config['num_time_interval']
        self.seed = config['seed']

        self.fit_n_iter = config['fit_n_iter']
        self.fit_bs = config['fit_bs']
        self.fit_thereshold = config['fit_thereshold']
        
        self.train_dropout = config['train_dropout']
        self.train_hiddens = config['train_hiddens']
        self.train_bs = config['train_bs']
        self.train_epoch = config['train_epoch']
        self.train_lr = config['train_lr']
        
        if 'lamb' in config:
            self.lamb = config['lamb']
            
        if 'sigma_rate' in config:
            self.sigma_rate = config['sigma_rate']

        if 'constraint_type' in config:
            self.c_type = config['constraint_type']
        else:
            self.c_type = None
            
        if 'feat_size' in config:
            self.feat_size = config['feat_size']
            self.feat_depth = config['feat_depth']
        else:
            self.feat_size = 10
            self.feat_depth = 2


class HeatEquation():
    """
    Simple isotropic Heat Equation in d-dimensions:
        ∂u/∂t = α * Δu
    where Δu is the Laplacian of u.

    This class defines the SDE-based simulation and PDE right-hand side for training.
    """

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = 0.0  # center initialization
        self.alpha = 0.1   # diffusion coefficient

    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE (Brownian motion) for heat diffusion.

        Returns:
            dw_sample: noise samples of shape (num_sample, dim, num_time_interval)
            x_sample: simulated paths of shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)
        x_sample[:, :, 0] = torch.ones(num_sample, self.dim, device=device) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f(self, t, x, y, z):
        """
        No reaction term for heat equation.
        Returns:
            Zero tensor of same shape as y.
        """
        return torch.zeros_like(y)

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition u(T, x), defined as:
            u(T, x) = g_param * exp(-|x|^2)
        """
        return g_param * torch.exp(-torch.norm(x, dim=-1, keepdim=True) ** 2)

    def rhs(self, u, theta_rep, x, t=0.2):
        """
        Right-hand side of the PDE:
            rhs = α * Δu
        """
        with torch.enable_grad():
            x.requires_grad_()
            u_val = u.forward_2(theta_rep, x).sum()
            grad_u = torch.autograd.grad(u_val, x, create_graph=True)[0]  # shape = (bs, n_x, dim)
            laplacian = 0.0
            for i in range(x.shape[-1]):
                grad_u_i = grad_u[..., i].sum()
                d2u_dxi2 = torch.autograd.grad(grad_u_i, x, create_graph=True)[0][..., i]
                laplacian += d2u_dxi2
        return self.alpha * laplacian.detach()

            
class PricingDefaultRisk():
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.lamb =0.01
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.rand(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)
        x_sample[:, :, 0] = torch.ones(num_sample, self.dim, device=device) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f(self, t, x, y, z):
        piecewise_linear = F.relu(
            F.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g(self, t, x, g_param=1.0):
        return g_param * torch.min(x, -1)[0]
    
    def rhs(self, u, theta_rep, x, t=0.2):
        """
        Compute the right-hand side of the PDE.

        Inputs:
            u: neural network representing the solution of the PDE
            theta_rep: torch.Tensor, shape (batch_size, num_points, n_params), parameters of u
            x: torch.Tensor, shape (batch_size, num_points, dim), spatial sampling points

        Outputs:
            rhs_: torch.Tensor, shape (batch_size, num_points)
        """
        u_ = u.forward_2(theta_rep, x) # shape=(bs, n_x)
        f_ = self.f(None, None, u_, None) # shape=(bs, n_x)
        rhs_ = f_  #the 3rd term of rhs

        with torch.enable_grad():
            x.requires_grad_()
            du_dx = torch.autograd.grad(u.forward_2(theta_rep,x).sum(), x, create_graph=True)[0] # shape=x.shape= (bs, n_x, d)
            rhs_ += self.mu_bar * (du_dx * x).sum(-1).detach() # shape=(bs, n_x), the 1st term of rhs
            du_dx = du_dx.sum(dim=(0,1)) #shape=(d,)

            for i in range(x.shape[-1]):
                du_dxi_dxi =  torch.autograd.grad(du_dx[i], x, create_graph=True)[0][...,i].detach() # shape=(bs, n_x)
                xi = x[...,i] # shape=(bs, n_x)
                rhs_ += 0.5 * self.sigma**2 * xi**2 * du_dxi_dxi   #the 2nd term of rhs
        
        return rhs_


class PricingDiffRate():
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending.
    """
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = 100.0
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample, device):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.normal(mean=0, std=self.sqrt_delta_t, size=(num_sample, self.dim, self.num_time_interval), device=device)
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)
        x_sample[:, :, 0] = torch.ones(num_sample, self.dim, device=device) * self.x_init

        factor = torch.tensor(np.exp((self.mu_bar - (self.sigma ** 2) / 2) * self.delta_t), device=device)
        
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * torch.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]

        return dw_sample, x_sample

    def f(self, t, x, y, z):
        temp = torch.sum(z, 2, keepdims=False) / self.sigma
        zero_tensor = torch.tensor(0.0, device=y.device)
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
            (self.rb - self.rl) * torch.max(temp - y, zero_tensor))

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition of the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)

        Outputs:
            g_value: torch.Tensor, shape (batch_size, num_points)
        """
        temp = torch.max(x, dim=2, keepdim=True).values  # shape=(bs, nx, 1)

        g_param = g_param.unsqueeze(2)
        output = (torch.max(temp - 120, torch.tensor(0.0)) - torch.max(temp - 150, torch.tensor(0.0))) * g_param
        return output.squeeze(-1)  # shape=(bs, nx)

    def rhs(self, u, theta_rep, x, t=0.2):
        ''' rhs of the PDE (time inversed of Eq.11 of PNAS paper)
        Inputs:
            u: the neural network that represents the solution of the PDE
            theta_rep: torch tensor, (bs, n_x, n_params) the params of u.
            x: spacial sampling points, (bs, n_x, dim)
        Outputs:
            rhs_: torch tensor, (bs, n_x) the right hand side of the PDE eval at x.
        '''
        u_ = u.forward_2(theta_rep, x) # shape=(bs, n_x)

        with torch.enable_grad():
            x.requires_grad_()
            du_dx = torch.autograd.grad(u.forward_2(theta_rep,x).sum(), x, create_graph=True)[0] # shape=x.shape= (bs, n_x, d)
            f_ = self.f(None, None, u_, du_dx) # shape=(bs, n_x)
            rhs_ = f_  #the 3rd term of rhs
            rhs_ += self.mu_bar * (du_dx * x).sum(-1).detach() # shape=(bs, n_x), the 1st term of rhs
            du_dx = du_dx.sum(dim=(0,1)) #shape=(d,)

            for i in range(self.dim):
                du_dxi_dxi =  torch.autograd.grad(du_dx[i], x, create_graph=True)[0][...,i].detach() # shape=(bs, n_x)
                xi = x[...,i] # shape=(bs, n_x)
                rhs_ += 0.5 * self.sigma**2 * xi**2 * du_dxi_dxi   #the 2nd term of rhs
        
        return rhs_


class BurgersType():
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = 0.0
        self.lamb = 2.0
        self.sigma = eqn_config.sigma_rate * self.dim + 0.0  # sigma rate

    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t
        # initialize x_sample
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)
        x_sample[:, :, 0] = self.x_init

        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]

        return dw_sample, x_sample

    def f(self, t, x, y, z):
        """
        Generator function in the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)
            y: torch.Tensor, shape (batch_size, num_points)
            z: torch.Tensor, shape (batch_size, num_points, dim) (unused here)

        Outputs:
            f_value: torch.Tensor, shape (batch_size, num_points)
        """
        if z is None:
            return torch.zeros_like(y)
        sum_z = torch.sum(z, dim=2)  # (batch_size, num_points)
        constant = (2 + self.dim) / (2.0 * self.dim)
        f_value = (y - constant) * sum_z  # (batch_size, num_points)
        return f_value

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition of the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)

        Outputs:
            g_value: torch.Tensor, shape (batch_size, num_points)
        """
        sum_x = torch.sum(x, dim=2).unsqueeze(-1)  # (batch_size, num_points, 1)
        g_param = g_param.unsqueeze(2)
        exponent = (t * g_param + sum_x / self.dim).squeeze(-1)
        g_value = 1 - 1.0 / (1 + torch.exp(exponent))  # (batch_size, num_points)
        return g_value

    def rhs(self, u, theta_rep, x, t=0.2):
        """
        input：
            u: the neural network that represents the solution of the PDE
            theta_rep: (batch_size, num_points, n_params)s
            x: (batch_size, num_points, dim)，sample point

        output：
            rhs_: (batch_size, num_points)
        """
        batch_size, num_points, dim = x.shape

        u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)

        # initialize rhs_ (may not used)
        rhs_ = -self.f(None, x, u_, None)  # (batch_size, num_points)

        with torch.enable_grad():
            x = x.detach()
            x.requires_grad_(True)

            u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)
            du_dx = torch.autograd.grad(u_.sum(), x, create_graph=True)[0]  # (batch_size, num_points, dim)

            # f(u, du/dx)
            f_ = self.f(None, x, u_, du_dx)  # (batch_size, num_points)
            rhs_ = -f_.detach()

            laplacian = torch.zeros_like(u_)  # (batch_size, num_points)
            for i in range(dim):
                du_dxi = du_dx[..., i]  # (batch_size, num_points)
                du_dxi_sum = du_dxi.sum()  # (1)

                du_dxi_dxi = torch.autograd.grad(du_dxi_sum, x, create_graph=True)[0][..., i]  # (batch_size, num_points)
                du_dxi_dxi = du_dxi_dxi.detach()
                laplacian += du_dxi_dxi

            diffusion_term = 0.5 * self.sigma ** 2 * laplacian  # (batch_size, num_points)
            rhs_ -= diffusion_term

        return rhs_


class ReactionDiffusion():
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)

        self._kappa = 0.6
        self.lamb = 5.0
        self.lambd = 1 / math.sqrt(self.dim)
        self.x_init = torch.zeros(self.dim)  # (dim,)
        self.y_init = None
        # self.y_init = 1 + self._kappa + torch.sin(self.lambd * torch.sum(self.x_init)) * torch.exp(
        #     -self.lambd * self.lambd * self.dim * self.total_time / 2)
    
    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t  # (num_sample, dim, num_time_interval)
        
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)  # (num_sample, dim, num_time_interval + 1)
        x_sample[:, :, 0] = self.x_init  # (num_sample, dim)
        
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        
        return dw_sample, x_sample

    def f(self, t, x, y, z):
        """
        Generator function in the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)
            y: torch.Tensor, shape (batch_size, num_points)
            z: torch.Tensor, shape (batch_size, num_points, dim) (unused here)

        Outputs:
            f_value: torch.Tensor, shape (batch_size, num_points)
        """
        # calculate exp_term and sin_term
        exp_term = torch.exp((self.lambd ** 2) * self.dim * (t - self.total_time) / 2)  # (batch_size, num_points)
        sum_x = torch.sum(x, dim=2)  # (batch_size, num_points)
        sin_term = torch.sin(self.lambd * sum_x)  # (batch_size, num_points)
        temp = y - self._kappa - 1 - sin_term * exp_term  # (batch_size, num_points)
        f_value = torch.minimum(torch.tensor(1.0, device=y.device), temp ** 2)  # (batch_size, num_points)
        return f_value

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition of the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)

        Outputs:
            g_value: torch.Tensor, shape (batch_size, num_points)
        """
        sum_x = torch.sum(x, dim=2)  # (batch_size, num_points)
        g_value = 1 + self._kappa + torch.sin(self.lambd * sum_x) * g_param  # (batch_size, num_points)
        return g_value

    def rhs(self, u, theta_rep, x, t=0.2):
        """
        Compute the right-hand side of the PDE.

        Inputs:
            u: neural network representing the solution of the PDE
            theta_rep: torch.Tensor, shape (batch_size, num_points, n_params), parameters of u
            x: torch.Tensor, shape (batch_size, num_points, dim), spatial sampling points

        Outputs:
            rhs_: torch.Tensor, shape (batch_size, num_points)
        """
        batch_size, num_points, dim = x.shape

        u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)

        rhs_ = self.f(t, x, u_, None)  # (batch_size, num_points)

        with torch.enable_grad():
            x = x.detach()
            x.requires_grad_(True)

            u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)

            du_dx = torch.autograd.grad(u_.sum(), x, create_graph=True)[0]  # (batch_size, num_points, dim)

            laplacian = torch.zeros_like(u_)  # (batch_size, num_points)
            for i in range(dim):
                du_dxi = du_dx[..., i]  # (batch_size, num_points)
                du_dxi_sum = du_dxi.sum()

                du_dxi_dxi = torch.autograd.grad(du_dxi_sum, x, create_graph=True)[0][..., i]  # (batch_size, num_points)
                laplacian += du_dxi_dxi

            diffusion_term = 0.5 * laplacian  # (batch_size, num_points)
            rhs_ -= diffusion_term

        return rhs_


class HJBLQ():
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = torch.zeros(self.dim)  # (dim,)
        self.sigma = math.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t  # (num_sample, dim, num_time_interval)
        
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)  # (num_sample, dim, num_time_interval + 1)
        x_sample[:, :, 0] = self.x_init 
        
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        
        return dw_sample, x_sample

    def f(self, t, x, y, z):
        """
        Generator function in the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)
            y: torch.Tensor, shape (batch_size, num_points)
            z: torch.Tensor, shape (batch_size, num_points, dim) (unused here)

        Outputs:
            f_value: torch.Tensor, shape (batch_size, num_points)
        """
        sum_z_square = torch.sum(z ** 2, dim=2)  # (batch_size, num_points)
        f_value = self.lambd * sum_z_square / 2  # (batch_size, num_points)
        return f_value

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition of the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)

        Outputs:
            g_value: torch.Tensor, shape (batch_size, num_points)
        """
        sum_x_square = g_param * torch.sum(x ** 2, dim=2)  # (batch_size, num_points)
        g_value = torch.log((1 + sum_x_square) / 2)  # (batch_size, num_points)
        return g_value

    def rhs(self, u, theta_rep, x, t):
        """
        Compute the right-hand side of the PDE.

        Inputs:
            u: neural network representing the solution of the PDE
            theta_rep: torch.Tensor, shape (batch_size, num_points, n_params), parameters of u
            x: torch.Tensor, shape (batch_size, num_points, dim), spatial sampling points

        Outputs:
            rhs_: torch.Tensor, shape (batch_size, num_points)
        """
        batch_size, num_points, dim = x.shape

        u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)

        with torch.enable_grad():
            x = x.detach()
            x.requires_grad_(True)

            u_ = u.forward_2(theta_rep, x)  # (batch_size, num_points)

            du_dx = torch.autograd.grad(u_.sum(), x, create_graph=True)[0]  # (batch_size, num_points, dim)

            # f(u, du/dx)
            f_ = self.f(t, x, u_, du_dx)  # (batch_size, num_points)
            rhs_ = f_  
            
            laplacian = torch.zeros_like(u_)  # (batch_size, num_points)
            for i in range(dim):
                du_dxi = du_dx[..., i]  # (batch_size, num_points)
                du_dxi_sum = du_dxi.sum()

                du_dxi_dxi = torch.autograd.grad(du_dxi_sum, x, create_graph=True)[0][..., i]  # (batch_size, num_points)
                laplacian += du_dxi_dxi

            diffusion_term = 0.5 * self.sigma ** 2 * laplacian  # (batch_size, num_points)
            rhs_ -= diffusion_term

        return rhs_


class AllenCahn():
    """
    Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim  # Dimension of the PDE
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = math.sqrt(self.delta_t)
        self.y_init = None

        self.x_init = torch.zeros(self.dim)  # Shape: (dim,)
        self.sigma = math.sqrt(2.0)

    def sample(self, num_sample, device, generator=None):
        """
        Sample forward SDE.

        Inputs:
            num_sample: int, number of samples in the batch
            device: torch.device, device to place tensors
            generator: torch.Generator, random number generator (optional)

        Outputs:
            dw_sample: torch.Tensor, shape (num_sample, dim, num_time_interval)
            x_sample: torch.Tensor, shape (num_sample, dim, num_time_interval + 1)
        """
        # Generate Brownian increments
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=device, generator=generator) * self.sqrt_delta_t  # Shape: (num_sample, dim, num_time_interval)

        # Initialize x_sample
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=device)  # Shape: (num_sample, dim, num_time_interval + 1)
        x_sample[:, :, 0] = self.x_init  # Set initial condition, shape: (dim,)

        # Simulate x over time
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]

        return dw_sample, x_sample

    def f(self, t, x, y, z):
        """
        Generator function in the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)
            y: torch.Tensor, shape (batch_size, num_points)
            z: torch.Tensor, shape (batch_size, num_points, dim) (unused here)

        Outputs:
            f_value: torch.Tensor, shape (batch_size, num_points)
        """
        f_value = y - y ** 3  # Shape: (batch_size, num_points)
        return f_value

    def g(self, t, x, g_param=1.0):
        """
        Terminal condition of the PDE.

        Inputs:
            t: time (unused here)
            x: torch.Tensor, shape (batch_size, num_points, dim)

        Outputs:
            g_value: torch.Tensor, shape (batch_size, num_points)
        """
        sum_square_x = torch.sum(x ** 2, dim=2)  # Shape: (batch_size, num_points)
        denom = 1 + 0.2 * sum_square_x  # Shape: (batch_size, num_points)
        g_value = (0.5 * g_param) / denom  # Shape: (batch_size, num_points)
        return g_value

    def rhs(self, u, theta_rep, x, t):
        """
        Compute the right-hand side of the PDE.

        Inputs:
            u: neural network representing the solution of the PDE
            theta_rep: torch.Tensor, shape (batch_size, num_points, n_params), parameters of u
            x: torch.Tensor, shape (batch_size, num_points, dim), spatial sampling points

        Outputs:
            rhs_: torch.Tensor, shape (batch_size, num_points)
        """
        batch_size, num_points, dim = x.shape

        # Ensure x requires gradient
        x = x.detach()
        x.requires_grad_(True)

        # Evaluate u at x
        u_ = u.forward_2(theta_rep, x)  # Shape: (batch_size, num_points)

        # Compute the reaction term f(u)
        f_ = self.f(None, x, u_, None)  # Shape: (batch_size, num_points)

        # Initialize rhs_ with f(u)
        rhs_ = f_.clone()  # Shape: (batch_size, num_points)

        with torch.enable_grad():
            # Since x requires grad, and u depends on x, we can compute du/dx
            # Recompute u_ to ensure correct computation graph
            u_ = u.forward_2(theta_rep, x)  # Shape: (batch_size, num_points)

            # Compute gradient du/dx
            du_dx = torch.autograd.grad(u_.sum(), x, create_graph=True)[0]  # Shape: (batch_size, num_points, dim)

            # Compute Laplacian (sum of second derivatives)
            laplacian = torch.zeros_like(u_)  # Shape: (batch_size, num_points)
            for i in range(dim):
                du_dxi = du_dx[:, :, i]  # Shape: (batch_size, num_points)

                # Compute second derivative w.r.t x_i
                du_dxi_dxi = torch.autograd.grad(du_dxi.sum(), x, create_graph=True)[0][:, :, i]  # Shape: (batch_size, num_points)

                laplacian += du_dxi_dxi  # Sum over dimensions

            # Compute diffusion term
            diffusion_term = 0.5 * laplacian  # Shape: (batch_size, num_points)

            # Update rhs_: rhs = f(u) - 0.5 * Laplacian(u)
            rhs_ = rhs_ - diffusion_term  # Shape: (batch_size, num_points)

        return rhs_

