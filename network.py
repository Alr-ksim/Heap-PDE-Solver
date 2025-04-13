import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# --------------------

class FFNet(torch.nn.Module):
    # --- copied from neural control codes
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        '''
        Inputs:
            d: input dims
            dd: output dims
            width: hidden dims,
            depth: num hiddens
        '''
        super(FFNet, self).__init__()
        is_mlp = not kwargs.get('mlp', False)
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.d = d
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.linearIn = nn.Linear(d, width, bias=is_mlp)

        # create hidden layers
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))

        # output layer is linear
        self.linearOut = nn.Linear(width, dd,bias = False)
        self.activation = activation_func

    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        # compute from i to i+1 layer
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp = self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)


class ResNet(torch.nn.Module):
    # --- copied from neural control codes
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        super(ResNet, self).__init__()
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        is_mlp = not kwargs.get('mlp', False)
        self.d = d
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.linearIn = nn.Linear(d, width, bias=is_mlp)

        # create hidden layers
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))

        # output layer is linear
        self.linearOut = nn.Linear(width, dd,bias = False)
        self.activation = activation_func
    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        # x = self.linearIn(x) # Match dimension
        # compute from i to i+1 layer
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp =  x + self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)
    
# --- U network ---
class U(nn.Module):
    def __init__(self, dim = 10, width=80):
        super().__init__()
        self.dim = dim
        self.width = width

        # Papemeters, Section 5.2.1. in the paper.
        # beta \in R^d
        # a \in R^{d*width}
        # b \in R^{width}
        # c \in R^{width}
        
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width))
        self.beta = nn.Parameter(torch.randn(self.dim))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width]]
        self.connections = ['F', 'F', 'E']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, beta = self.a, self.b, self.c, self.beta
        return torch.tanh(torch.sin(math.pi * (x- beta)) @ a - b) @ c
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        

        u0 = torch.sin(math.pi * (x- beta)) #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)


class U_HJB(nn.Module):
    def __init__(self, dim = 8, width=50):
        super().__init__()
        self.dim = dim
        self.width = width

        # Papemeters, Section 5.2.3. in the paper.
        # a \in R^{d*width}
        # b \in R^{d*width}
        # w \in R^{width}
        self.c = nn.Parameter(torch.randn(self.dim)) #add a bias layer to make the graph connected.
        self.b = nn.Parameter(torch.randn(self.dim,self.width))
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.w = nn.Parameter(torch.randn(self.width)) 
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [dim, width], [width]]
        self.connections = ['F', 'E', 'F', ]
    
    def forward(self, x):
        '''
        Inputs:
            x: torch.Tensor, shape (bs, dim).  bs denotes batch size.
        Outputs:
            u: torch.Tensor, shape (bs)
        '''
        a, b, w, c = self.a, self.b, self.w, self.c
        a = a.unsqueeze(0) # shape = (1,  dim, width)
        b = b.unsqueeze(0) # shape = (1,  dim, width)
        c = c.unsqueeze(0).unsqueeze(2) # shape = (1, dim, 1)
        x = x.unsqueeze(2) # shape = (bs, dim, 1)

        out = a * (x - c - b) # shape = (bs, dim, width)
        out = torch.linalg.vector_norm(out, dim=-2) # shape = (bs, width)
        out = torch.exp(-out/2) # shape = (bs, width)
        out = torch.mv(out, w) #(bs, width)*(width,) --> (bs,)
        out = out
        return out 
    
    def forward_2(self, theta, x):
        '''
        Inputs:
            theta: torch.Tensor, shape (bs, n_x,  n_params) or (bs, n_params)
            x: torch.Tensor, shape (bs, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (bs, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        c_size = dim 
        a_size = dim * width
        b_size = dim * width
        w_size = width
        

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        # parameters must be ordered as computation in forward function.
        idx = 0
        c = theta[..., :c_size].reshape(bs, nx, dim, 1)
        idx += c_size
        b = theta[..., idx:idx+b_size].reshape(bs, nx, dim, width)
        idx += b_size
        a = theta[..., idx:idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        w = theta[..., idx:idx+w_size]  #shape=(bs, nx, width)
        
        x = x.unsqueeze(-1) # shape = (bs, n_x, d, 1)

        #print(a.shape, b.shape, w.shape, x.shape)
        out = a * (x -c - b) # shape = (bs, n_x, d, width)
        out = torch.linalg.vector_norm(out, dim=-2) # shape = (bs, n_x, width)
        out = torch.exp(-out/2) # shape =  (bs, n_x, width)
        out = (out * w).sum(-1) # shape = (bs, n_x)
        out = out 
        return out


class U_PRICE(nn.Module):
    def __init__(self, dim = 10, width=80):
        super().__init__()
        self.dim = dim
        self.width = width

        self.beta = nn.Parameter(torch.randn(self.dim))
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width]]
        self.connections = ['F', 'F', 'E']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, beta = self.a, self.b, self.c, self.beta
        return torch.tanh((x- beta) @ a - b) @ c
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        Tode:
            fit the g function from theta and x
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        
        u0 = x- beta #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)


class U_TWO(nn.Module):
    def __init__(self, input_dim=10, hidden_dim1=50, hidden_dim2=50, output_dim=1): # burgers: 10 80 60 1
        super().__init__()
        self.dim = input_dim
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        # Define learnable parameters
        self.W1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim1))   # (input_dim, hidden_dim1)
        self.b1 = nn.Parameter(torch.randn(self.hidden_dim1))                   # (hidden_dim1,)
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim2)) # (hidden_dim1, hidden_dim2)
        self.b2 = nn.Parameter(torch.randn(self.hidden_dim2))                   # (hidden_dim2,)
        self.W3 = nn.Parameter(torch.randn(self.hidden_dim2, self.output_dim))  # (hidden_dim2, output_dim)
        self.b3 = nn.Parameter(torch.randn(self.output_dim))                    # (output_dim,)

        self.n_params = sum(p.numel() for p in self.parameters())

        # Define the dimensions of the parameters
        self.dims = [
            [self.input_dim, self.hidden_dim1],    # for W1
            [self.hidden_dim1],                    # for b1
            [self.hidden_dim1, self.hidden_dim2],  # for W2
            [self.hidden_dim2],                    # for b2
            [self.hidden_dim2, self.output_dim],   # for W3
            [self.output_dim]                      # for b3
        ]

        self.connections = ['F', 'F', 'F', 'F', 'F']  # W1->b1, b1->W2, W2->b2, b2->W3, W3->b3

    def forward(self, x):
        ''' Single forward
        Input:
            x: torch.Tensor, shape (batch_size, input_dim)
        Output:
            y: torch.Tensor, shape (batch_size, output_dim)
        '''
        # Calculate the first layer
        # x shape: (batch_size, input_dim)
        # W1 shape: (input_dim, hidden_dim1)
        # b1 shape: (hidden_dim1,)
        z1 = x @ self.W1 + self.b1  # shape: (batch_size, hidden_dim1)
        hidden1 = torch.tanh(z1)    # shape: (batch_size, hidden_dim1)

        # Calculate the second layer
        # hidden1 shape: (batch_size, hidden_dim1)
        # W2 shape: (hidden_dim1, hidden_dim2)
        # b2 shape: (hidden_dim2,)
        z2 = hidden1 @ self.W2 + self.b2  # shape: (batch_size, hidden_dim2)
        hidden2 = torch.tanh(z2)          # shape: (batch_size, hidden_dim2)

        # Calculate the output layer
        # hidden2 shape: (batch_size, hidden_dim2)
        # W3 shape: (hidden_dim2, output_dim)
        # b3 shape: (output_dim,)
        output = hidden2 @ self.W3 + self.b3  # shape: (batch_size, output_dim)

        return output

    def forward_2(self, theta, x):
        ''' Batch forward
        Input:
            theta: torch.Tensor, shape (batch_size, n_x, len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, input_dim) or (n_x, input_dim)
        Output:
            y: torch.Tensor, shape (batch_size, n_x, output_dim)
        '''
        bs = theta.shape[0]  # batch_size
        nx = x.shape[1] if x.dim() > 2 else x.shape[0]  # n_x
        input_dim = self.input_dim
        hidden_dim1 = self.hidden_dim1
        hidden_dim2 = self.hidden_dim2
        output_dim = self.output_dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  # shape: (bs, nx, len_theta)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)  # shape: (bs, nx, input_dim)

        W1_size = input_dim * hidden_dim1
        b1_size = hidden_dim1
        W2_size = hidden_dim1 * hidden_dim2
        b2_size = hidden_dim2
        W3_size = hidden_dim2 * output_dim
        b3_size = output_dim

        # Get the parameters from theta
        idx = 0
        W1 = theta[..., idx: idx + W1_size].reshape(bs, nx, input_dim, hidden_dim1)
        idx += W1_size
        b1 = theta[..., idx: idx + b1_size].reshape(bs, nx, 1, hidden_dim1)
        idx += b1_size
        W2 = theta[..., idx: idx + W2_size].reshape(bs, nx, hidden_dim1, hidden_dim2)
        idx += W2_size
        b2 = theta[..., idx: idx + b2_size].reshape(bs, nx, 1, hidden_dim2)
        idx += b2_size
        W3 = theta[..., idx: idx + W3_size].reshape(bs, nx, hidden_dim2, output_dim)
        idx += W3_size
        b3 = theta[..., idx: idx + b3_size].reshape(bs, nx, 1, output_dim)

        # x shape: (bs, nx, input_dim)
        x_expanded = x.unsqueeze(-2)  # shape: (bs, nx, 1, input_dim)

        # Calculate the first layer
        # W1 shape: (bs, nx, input_dim, hidden_dim1)
        # b1 shape: (bs, nx, 1, hidden_dim1)
        z1 = x_expanded @ W1 + b1  # shape: (bs, nx, 1, hidden_dim1)
        hidden1 = torch.tanh(z1)  # shape: (bs, nx, 1, hidden_dim1)

        # Calculate the second layer
        # W2 shape: (bs, nx, hidden_dim1, hidden_dim2)
        # b2 shape: (bs, nx, 1, hidden_dim2)
        z2 = hidden1 @ W2 + b2  # shape: (bs, nx, 1, hidden_dim2)
        hidden2 = torch.tanh(z2)  # shape: (bs, nx, 1, hidden_dim2)

        # Calculate the output layer
        # W3 shape: (bs, nx, hidden_dim2, output_dim)
        # b3 shape: (bs, nx, 1, output_dim)
        output = hidden2 @ W3 + b3  # shape: (bs, nx, 1, output_dim)

        output = output.squeeze(-2).squeeze(-1)  # shape: (bs, nx)
        return output


class V(nn.Module):
    r"""V network in the paper, mapping from theta to dtheta_dt."""
    def __init__(self, dim = 970, width=1000, depth=5, dropout_p=0, activation_func=F.relu):
        super(V, self).__init__()
        self.dim = dim
        self.width = width

        # Papemeters, Section 5.1. in the paper.
        self.S = FFNet(dim, 1, width, depth,  F.sigmoid, )  
        self.R = ResNet(dim, dim, width, depth,  activation_func, )
        self.E = FFNet(dim, dim, width, depth,  activation_func, )

        self.dropout_p = dropout_p
    
    def forward(self, theta, generator=None):
        '''
        Inputs:
            theta: torch.Tensor, shape (batch_size, dim)
        Outputs:
            dtheta_dt: torch.Tensor, shape (batch_size, dim)
        '''
        S = self.S(theta)
        R = self.R(theta)
        E = self.E(theta)
        
        dtheta_dt = S * (R + E*theta)
        dtheta_dt = dropout(dtheta_dt, p=self.dropout_p, training=self.training, g=generator) #add drop out
            
        return dtheta_dt


def dropout(x, p, g, training):
    r""" iid dropout + rescale of tensor.
    Inputs:
        x: torch.Tensor, shape (batch_size, dim)
        p: float, dropout rate
        g: torch.Generator
        training: bool, whether to apply dropout
    Outputs:
        x: torch.Tensor, shape (batch_size, dim)
    """
    if p > 0 and training:
        mask = torch.bernoulli((1-p)*torch.ones_like(x), generator=g)
        x = x * mask / (1-p)
    return x

# --- Adaptive PDQP network BEGIN ---
def initialize_linear_layers(model, init_fn='normal', mean=0.0, std=0.01, a=-0.01, b=0.01):
    """
    Initialize all Linear layers in the given model with the specified initialization method.
    
    Args:
        model (torch.nn.Module): The model whose linear layers need to be initialized.
        init_fn (str): The initialization method. Can be 'normal', 'uniform', 'xavier', or 'he'.
        mean (float): Mean for normal distribution initialization (only used for 'normal').
        std (float): Standard deviation for normal distribution initialization (only used for 'normal').
        a (float): Lower bound for uniform distribution initialization (only used for 'uniform').
        b (float): Upper bound for uniform distribution initialization (only used for 'uniform').
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f'Initializing {name} with {init_fn} initialization.')

            if init_fn == 'normal':
                nn.init.normal_(module.weight, mean=mean, std=std)
            elif init_fn == 'uniform':
                nn.init.uniform_(module.weight, a=a, b=b)
            elif init_fn == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_fn == 'he':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                raise ValueError(f"Unsupported initialization method: {init_fn}")
            
            # Optionally, initialize bias to zero (you can modify this behavior)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class Adaptive_PDQP_Net(torch.nn.Module):
    '''AN EFFICIENT UNSUPERVISED FRAMEWORK FOR CONVEX QUADRATIC PROGRAMS VIA DEEP UNROLLING
    Algo Ref: Algorithm 2. PDQP-Net
    Code Ref: https://github.com/CenturionMKIII/PDHG-Net-New/blob/main/src/model.py
    '''
    def __init__(self, feat_sizes, dropout_p):
        '''
        Input:
            feat_sizes: list of int, sizes of the hidden layers
            NOTE: x_size and y_size should not be used as size of trainable parameters, otherwise the model cannot generalize to unseen sizes.
        '''
        super().__init__()
        
        self.dropout_p = dropout_p
        
        # f_x and f_y MLPs
        self.cons_embedding = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_sizes[0], bias=True),
            #nn.ReLU()
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(in_features=1, out_features=feat_sizes[0], bias=True),
            #nn.ReLU()
        )

        # K PDHG iteration layers
        self.layers = nn.ModuleList()
        for indx in range(len(feat_sizes)-1):
            self.layers.append(Adaptive_PDQP_Layer(feat_sizes[indx], feat_sizes[indx+1]))

        # g_x and g_y MLPs
        self.output_module1 = nn.Sequential(
            #nn.Linear(in_features=feat_sizes[-1], out_features=feat_sizes[-1], bias=True),
            #nn.ReLU(),
            nn.Linear(in_features=feat_sizes[-1], out_features=1, bias=False)
        )
        self.output_module2 = nn.Sequential(
            #nn.Linear(in_features=feat_sizes[-1], out_features=feat_sizes[-1], bias=True),
            #nn.ReLU(),
            nn.Linear(in_features=feat_sizes[-1], out_features=1, bias=False)
        )

        initialize_linear_layers(self, init_fn='normal', mean=1.0, std=0.000,)

    def print_step_param(self):
        for index, layer in enumerate(self.layers):
            print(f"In layer {index}, eta = {layer.left.eta}, tau = {layer.right.tau}")
        
    def forward(self, x, y, Q, A, c, b, l=None, u=None, theta=None):
        '''
        Input:
            x: shape(bs, n), primal var initial guess
            y: shape(bs, m), dual var initial guess
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            b: shape(bs, m), constraint
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
            theta: shape(bs, n), input of initial guess network
        Output:
            x_pred: shape(bs, n), primal var solved
            y_pred: shape(bs, m), dual var solved
        '''
        # Embedding step
        x_incre = self.var_embedding(x.unsqueeze(-1))  # (bs, n, d)
        X = x.unsqueeze(-1) + x_incre # (bs, n, d)
        Y = self.cons_embedding(y.unsqueeze(-1))  # (bs, m, d)
        X_bar = torch.zeros_like(X) + X  # (bs, n, d)

        # Iterate through PDQP layers
        for index, layer in enumerate(self.layers):
            X, X_bar, Y = layer(X, X_bar, Y, Q, A, c, b, l, u)

        # Final predictions
        #x_pred = self.output_module1(X).squeeze(-1)  # (bs, n)
        x_pred = (X - x_incre).mean(-1)
        #assert torch.isclose(x_pred, x).all()
        y_pred = self.output_module1(Y).squeeze(-1)  # (bs, m)
        

        return x_pred, y_pred
         
class Adaptive_PDQP_Layer(torch.nn.Module):
    '''Interface for PDQP_Layer_X and PDQP_Layer_Y'''
    def __init__(self, in_size, out_size):
        super().__init__()
        self.left = Adaptive_PDQP_Layer_X(in_size, out_size)
        self.right = Adaptive_PDQP_Layer_Y(in_size, out_size)
        self.alpha = 0.95
        self.eta_a = 0.95
    
    def update_param(self, X, X_new, Y, Y_new, Q, A, c, b):
        X_layer = self.left
        Y_layer = self.right
        
        # Calculate residual
        p_res = (1/X_layer.eta)*(X - X_new) - torch.bmm(A.transpose(1, 2), (Y - Y_new))
        d_res = (1/Y_layer.tau)*(Y - Y_new) - torch.bmm(A, (X - X_new))
        p_res = torch.norm(p_res, p=2, dim=(1, 2)).mean()
        d_res = torch.norm(d_res, p=2, dim=(1, 2)).mean()
        
        # Calculate backtrack
        c_value = 0.9
        x_diff = X_new - X
        y_diff = Y_new - Y
        x_norm = (c_value/(2*X_layer.eta)) * torch.norm(x_diff, p=2, dim=(1, 2)) ** 2
        y_norm = (c_value/(2*Y_layer.tau)) * torch.norm(y_diff, p=2, dim=(1, 2)) ** 2
        inner_pro = 2 * torch.bmm(y_diff.transpose(1, 2), torch.bmm(A, x_diff)).sum(dim=(1, 2))
        backtrack = (x_norm - inner_pro + y_norm).mean()
        
        if backtrack <= 0:
            X_layer.eta = X_layer.eta/2
            Y_layer.tau = Y_layer.tau/2
        if 2*p_res < d_res:
            X_layer.eta = X_layer.eta * (1 - self.alpha)
            Y_layer.tau = Y_layer.tau / (1 - self.alpha)
            self.alpha = self.alpha * self.eta_a
        elif p_res > 2*d_res:
            X_layer.eta = X_layer.eta / (1 - self.alpha)
            Y_layer.tau = Y_layer.tau * (1 - self.alpha)
            self.alpha = self.alpha * self.eta_a

    def forward(self, X, X_bar, Y, Q, A, c, b, l, u):
        X_new, X_bar_new = self.left(X, X_bar, Y, Q, A, c, l, u)
        Y_new = self.right(X, X_new, Y, A, b)
        
        with torch.no_grad():
            self.update_param(X, X_new, Y, Y_new, Q, A, c, b)
            
        return X_new, X_bar_new, Y_new
    
class Adaptive_PDQP_Layer_X(torch.nn.Module):
    
    def __init__(self,in_size, out_size):
        super().__init__()
        self.Wkx = nn.Linear(in_features=in_size, out_features=out_size, bias=False)  #NOTE: PDHG uses default bias=True, inconsistent with the paper
        self.Wky = nn.Linear(in_features=in_size, out_features=out_size, bias=False)
        self.beta= nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.eta= 1e-5
        self.lr_act = nn.Sigmoid()
    
    def forward(self, X, X_bar, Y, Q, A, c, l, u):
        '''
        Input:
            X: shape(bs, n, d_in), primal var prev iter
            X_bar: shape(bs, n, d_in), primal var momentum prev iter
            Y: shape(bs, m, d_in), dual var prev iter
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
        Output:
            X_new: shape(bs, n, d_out), primal var next iter
            X_bar_new: shape(bs, n, d_out), primal var momentum next iter
        '''
        # eta = self.lr_act(self.eta) * 2e-2 #NOTE: eta must be samll for stability
        eta = self.eta
        beta = self.lr_act(self.beta) 
        n_x = X.shape[-2]
        X_m = (1-beta) * X_bar + beta * X  # (bs, n, d_in)

        #X_new = X  - eta *(Q @self.Wkx(X_m) + c.unsqueeze(-1) - A.transpose(-1, -2) @ self.Wky(Y)) # (bs, n, d_out)
        #NOTE: use torch.sparse.mm(A, ...) if A is sparse   
        #NOTE: implicitly assume d_in == d_out.
        X_new = X  - eta *(Q @self.Wkx(X_m) + c.unsqueeze(-1))/n_x #NOTE:1. divide n_x for stability, 2. remove the dual term for debugging

        # Compute binary indicators for lower and upper bounds
        X_new = torch.clamp(X_new, min=l.unsqueeze(-1), max=u.unsqueeze(-1))  # (bs, n, d_out)

        X_bar_new  = (1-beta) * X_bar + beta * X_new
        
        return X_new, X_bar_new
    
class Adaptive_PDQP_Layer_Y(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super().__init__()
        self.Ws = nn.Linear(in_features=in_size, out_features=out_size, bias=False)
        self.tau= 1e-5
        self.theta= nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.act = nn.ReLU()
        self.lr_act = nn.Sigmoid()
    def forward(self, X, X_new, Y, A, b):
        '''
        Input:
            X: shape(bs, n, d_in), primal var prev iter
            X_new: shape(bs, n, d_out), primal var next iter
            Y: shape(bs, m, d_in), dual var prev iter
            A: shape(bs, m, n), constraint
            b: shape(bs, m), constraint
        Output:
            Y_new: shape(bs, m, d_out), primal var next iter
        '''
        theta = self.lr_act(self.theta)
        # tau = self.lr_act(self.tau)
        tau = self.tau
        X_tmp = theta * (X_new - X) + X_new
        Y_new = Y + tau * (b.unsqueeze(-1) - A@ self.Ws(X_tmp)) # (bs, m, d_out), NOTE: implicitly assume d_in == d_out.
        Y_new = self.act(Y_new)
        return Y_new

# --- Adaptive PDQP network END ---

# --- PDQP_Net_2 BEGIN ---
class PDQP_Net_2(torch.nn.Module):
    ''' Compare to PDQP_Net, PDQP_Net_2 linear weights are output of network, not trainable parameters.
    '''
    def __init__(self, feat_sizes, dropout_p, u_n_params):
        '''
        Input:
            feat_sizes: list of int, sizes of the hidden layers
            NOTE: x_size and y_size should not be used as size of trainable parameters, otherwise the model cannot generalize to unseen sizes.
        '''
        super().__init__()
        
        self.dropout_p = dropout_p
        
        self.n_params = 0
        self.n_params += feat_sizes[0]

        self.layers = []
        for indx in range(len(feat_sizes)-1):
            self.layers.append(PDQP_Layer_X_2(feat_sizes[indx], feat_sizes[indx+1]))
            self.n_params += self.layers[-1].n_params

        self.net = ResNet(u_n_params, self.n_params, u_n_params, 5, activation_func=nn.ReLU(), use_dropout=True)
        self.feat_sizes = feat_sizes


    def forward(self, x, y, Q, A, c, b, l=None, u=None, theta=None):
        '''
        Input:
            x: shape(bs, n), primal var initial guess
            y: shape(bs, m), dual var initial guess
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            b: shape(bs, m), constraint
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
            theta: shape(bs, n), input of initial guess network
        Output:
            x_pred: shape(bs, n), primal var solved
            y_pred: shape(bs, m), dual var solved
        '''
        params = self.net(theta)
        idx = 0
        var_emb_mat = params[:, :self.feat_sizes[0]].unsqueeze(1)  # (bs, 1, d)
        idx += self.feat_sizes[0]

        # Embedding step
        x_incre = x.unsqueeze(-1) @ var_emb_mat  # (bs, n, d)
        X = x.unsqueeze(-1) + x_incre # (bs, n, d)
        X_bar = torch.zeros_like(X) + X  # (bs, n, d)

        # Iterate through PDQP layers
        for index, layer in enumerate(self.layers):
            X, X_bar = layer(X, X_bar, None, Q, A, c, b, l, u, params, idx)
            idx += layer.n_params
 
        # Final predictions
        x_pred = (X - x_incre).mean(-1)
        y_pred = y
        
        return x_pred, y_pred
   
class PDQP_Layer_X_2(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super().__init__()
        self.lr_act = nn.Sigmoid()
        self.n_params = in_size * out_size + 2  # one matrix and two step_size
        self.in_size = in_size
        self.out_size = out_size
        self.alpha = 0.95
        self.eta_a = 0.95
        self.eta = 1e-5
    
    def update_param(self, X, X_new, A):
        # Calculate residual
        p_res = (1/self.eta)*(X - X_new)
        d_res = torch.bmm(A, (X - X_new))
        p_res = torch.norm(p_res, p=2, dim=(1, 2)).mean()
        d_res = torch.norm(d_res, p=2, dim=(1, 2)).mean()
        
        # Calculate backtrack no need ?
        c_value = 0.9
        x_diff = X_new - X
        x_norm = (c_value/(2*self.eta)) * torch.norm(x_diff, p=2, dim=(1, 2)) ** 2
        backtrack = (x_norm).mean()
        
        if backtrack <= 0:
            self.eta = self.eta/2
        if 2*p_res < d_res:
            self.eta = self.eta * (1 - self.alpha)
            self.alpha = self.alpha * self.eta_a
        elif p_res > 2*d_res:
            self.eta = self.eta / (1 - self.alpha)
            self.alpha = self.alpha * self.eta_a
                    
    def forward(self, X, X_bar, Y, Q, A, c, b, l, u, params, idx):
        '''
        Input:
            X: shape(bs, n, d_in), primal var prev iter
            X_bar: shape(bs, n, d_in), primal var momentum prev iter
            Y: shape(bs, m, d_in), dual var prev iter
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
        Output:
            X_new: shape(bs, n, d_out), primal var next iter
            X_bar_new: shape(bs, n, d_out), primal var momentum next iter
        '''
        eta = self.lr_act(params[:, idx:idx+1]).reshape(-1, 1, 1)  #NOTE: eta must be samll for stability
        # eta = self.eta
        beta = self.lr_act(params[:, idx+1:idx+2]).reshape(-1, 1, 1)        #shape=(bs, 1, 1)
        idx += 2
        Wkx = params[:, idx:idx+self.in_size*self.out_size].reshape(-1, self.in_size, self.out_size)
        Wkx = torch.softmax(Wkx, dim=1) # normalize


        n_x = X.shape[-2]
        d_in = X.shape[-1]
        X_m = (1-beta) * X_bar + beta * X  # (bs, n, d_in)

        #X_new = X  - eta *(Q @self.Wkx(X_m) + c.unsqueeze(-1) - A.transpose(-1, -2) @ self.Wky(Y)) # (bs, n, d_out)
        #NOTE: use torch.sparse.mm(A, ...) if A is sparse   
        #NOTE: implicitly assume d_in == d_out.
        X_new = X  - eta *(Q @ X_m @ Wkx + c.unsqueeze(-1))/(n_x * d_in) #NOTE:1. divide n_x for stability, 2. remove the dual term for debugging

        # Compute binary indicators for lower and upper bounds
        X_new = torch.clamp(X_new, min=l.unsqueeze(-1), max=u.unsqueeze(-1))  # (bs, n, d_out)

        X_bar_new  = (1-beta) * X_bar + beta * X_new
        
        # with torch.no_grad():
        #     self.update_param(X, X_new, A)

        return X_new, X_bar_new
    
# --- PDQP_Net_2 END ---

# --- PDQP_Net_2_full BEGIN ---
class PDQP_Net_2_full(torch.nn.Module):
    ''' Compare to PDQP_Net, PDQP_Net_2 linear weights are output of network, not trainable parameters.
    '''
    def __init__(self, feat_sizes, dropout_p, u_n_params):
        '''
        Input:
            feat_sizes: list of int, sizes of the hidden layers
            NOTE: x_size and y_size should not be used as size of trainable parameters, otherwise the model cannot generalize to unseen sizes.
        '''
        super().__init__()
        
        self.dropout_p = dropout_p
        
        self.var_emb_size = feat_sizes[0]
        self.cons_emb_size = feat_sizes[0]
        self.n_params = 0
        self.n_params += self.var_emb_size + self.cons_emb_size

        self.layers = []
        for indx in range(len(feat_sizes)-1):
            self.layers.append(PDQP_Layer_2_full(feat_sizes[indx], feat_sizes[indx+1]))
            self.n_params += self.layers[-1].n_params

        self.output_mod_size = feat_sizes[-1]
        self.n_params += self.output_mod_size
        self.net = ResNet(u_n_params, self.n_params, u_n_params, 5, activation_func=nn.ReLU(), use_dropout=True)
        self.feat_sizes = feat_sizes


    def forward(self, x, y, Q, A, c, b, l=None, u=None, theta=None):
        '''
        Input:
            x: shape(bs, n), primal var initial guess
            y: shape(bs, m), dual var initial guess
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            b: shape(bs, m), constraint
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
            theta: shape(bs, n), input of initial guess network
        Output:
            x_pred: shape(bs, n), primal var solved
            y_pred: shape(bs, m), dual var solved
        '''
        params = self.net(theta)
        idx = 0
        var_emb_mat = params[:, idx: idx+self.var_emb_size].unsqueeze(1)  # (bs, 1, d)
        idx += self.var_emb_size
        cons_emb_mat = params[:, idx: idx+self.cons_emb_size].unsqueeze(1)  # (bs, 1, d)
        idx += self.cons_emb_size

        # Embedding step
        x_incre = x.unsqueeze(-1) @ var_emb_mat  # (bs, n, d)
        X = x.unsqueeze(-1) + x_incre # (bs, n, d)
        X_bar = torch.zeros_like(X) + X  # (bs, n, d)
        Y = y.unsqueeze(-1) @ cons_emb_mat # (bs, m, d)

        # Iterate through PDQP layers
        for index, layer in enumerate(self.layers):
            X, X_bar, Y = layer(X, X_bar, Y, Q, A, c, b, l, u, params, idx)
            idx += layer.n_params
 
        # Final predictions
        output_mod = params[:, -self.output_mod_size:].unsqueeze(2)  # (bs, d, 1)
        
        x_pred = (X - x_incre).mean(-1) # (bs, n)
        y_pred = (Y @ output_mod).squeeze(-1) # (bs, m)

        return x_pred, y_pred

class PDQP_Layer_2_full(torch.nn.Module):
    '''Interface for PDQP_Layer_X and PDQP_Layer_Y'''
    def __init__(self, in_size, out_size):
        super().__init__()
        self.left = PDQP_Layer_X_2_full(in_size, out_size)
        self.right = PDQP_Layer_Y_2_full(in_size, out_size)
        self.n_params = self.left.n_params + self.right.n_params
        
    def forward(self,X, X_bar, Y, Q, A, c, b, l, u, params, idx):
        layer_idx = idx
        X_new, X_bar_new = self.left(X, X_bar, Y, Q, A, c, l, u, params, layer_idx)
        layer_idx += self.left.n_params
        Y_new = self.right(X, X_new, Y, A, b, params, layer_idx)
        return X_new, X_bar_new, Y_new
  
class PDQP_Layer_X_2_full(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super().__init__()
        self.lr_act = nn.Sigmoid()
        self.n_params = in_size * out_size + 2  # one matrix and two step_size
        self.in_size = in_size
        self.out_size = out_size
                    
    def forward(self, X, X_bar, Y, Q, A, c, l, u, params, idx):
        '''
        Input:
            X: shape(bs, n, d_in), primal var prev iter
            X_bar: shape(bs, n, d_in), primal var momentum prev iter
            Y: shape(bs, m, d_in), dual var prev iter
            Q: shape(bs, n, n), positive semi-definite
            A: shape(bs, m, n), constraint
            c: shape(bs, n), linear coefficent
            l: shape(bs, n), lower bound
            u: shape(bs, n), upper bound
        Output:
            X_new: shape(bs, n, d_out), primal var next iter
            X_bar_new: shape(bs, n, d_out), primal var momentum next iter
        '''
        eta = self.lr_act(params[:, idx:idx+1]).reshape(-1, 1, 1)  #NOTE: eta must be samll for stability
        beta = self.lr_act(params[:, idx+1:idx+2]).reshape(-1, 1, 1)        #shape=(bs, 1, 1)
        idx += 2
        Wkx = params[:, idx:idx+self.in_size*self.out_size].reshape(-1, self.in_size, self.out_size) # (bs, n1, n2)
        Wkx = torch.softmax(Wkx, dim=1) # normalize

        n_x = X.shape[-2]
        d_in = X.shape[-1]
        X_m = (1-beta) * X_bar + beta * X  # (bs, n, d_in)

        #X_new = X  - eta *(Q @self.Wkx(X_m) + c.unsqueeze(-1) - A.transpose(-1, -2) @ self.Wky(Y)) # (bs, n, d_out)
        #NOTE: use torch.sparse.mm(A, ...) if A is sparse   
        #NOTE: implicitly assume d_in == d_out.
        X_new = X  - eta *(Q @ X_m @ Wkx + c.unsqueeze(-1))/(n_x * d_in) #NOTE:1. divide n_x for stability, 2. remove the dual term for debugging

        # Compute binary indicators for lower and upper bounds
        X_new = torch.clamp(X_new, min=l.unsqueeze(-1), max=u.unsqueeze(-1))  # (bs, n, d_out)

        X_bar_new  = (1-beta) * X_bar + beta * X_new

        return X_new, X_bar_new

class PDQP_Layer_Y_2_full(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_params = in_size * out_size + 2 # Ws size + tau + theta
        self.act = nn.ReLU()
        self.lr_act = nn.Sigmoid()
        
    def forward(self, X, X_new, Y, A, b, params, idx):
        '''
        Input:
            X: shape(bs, n, d_in), primal var prev iter
            X_new: shape(bs, n, d_out), primal var next iter
            Y: shape(bs, m, d_in), dual var prev iter
            A: shape(bs, m, n), constraint
            b: shape(bs, m), constraint
        Output:
            Y_new: shape(bs, m, d_out), primal var next iter
        '''
        tau = self.lr_act(params[:, idx:idx+1]).reshape(-1, 1, 1)
        theta = self.lr_act(params[:, idx+1:idx+2]).reshape(-1, 1, 1) # (bs, 1, 1)
        idx += 2
        Ws = params[:, idx:idx+self.in_size*self.out_size].reshape(-1, self.in_size, self.out_size)
        Ws = torch.softmax(Ws, dim=1)
        
        X_tmp = theta * (X_new - X) + X_new
        Y_new = Y + tau * (b.unsqueeze(-1) - A @ X_tmp @ Ws) # (bs, m, d_out), NOTE: implicitly assume d_in == d_out.
        Y_new = self.act(Y_new)
        return Y_new

# --- PDQP_Net_2_full END ---
