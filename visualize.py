import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from network import U, U_HJB, V, U_TWO, Adaptive_PDQP_Net
from main import EqnConfig, inference, inference_2
from equation import PricingDefaultRisk, PricingDiffRate, BurgersType, ReactionDiffusion, HJBLQ, AllenCahn, HeatEquation

from utils import set_gpu_max_mem, set_seed

def visualize_pricing(eqn_config, test_num, rand_num):
    #--- load data ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = U_TWO(eqn_config.dim, hidden_dim1=40, hidden_dim2=40).to(device)
    dataset_test = torch.load(f'checkpoints/{eqn_config.eqn_type}_dataset_dim{u.dim}_params{u.n_params}_seed{eqn_config.seed}_rand.pt', weights_only=False)
    
    bs = 2
    rate = 1e-3
    theta_0 = dataset_test.tensors[0][rand_num*bs:(rand_num+1)*bs].to(device)
    mask = [True, False]
    theta_0 = theta_0[mask]
    bs = len(theta_0)
    T = eqn_config.total_time
    n_t = eqn_config.num_time_interval
    dropout_p = eqn_config.train_dropout
    c_type = eqn_config.c_type
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
    else:
        eqn = None

    #  --- define x, only first two dimensions non-zero ---- 
    N = 200
    a = 0.0 #left boundary
    b = 1.0 #right boundary
    x = np.linspace(a, b, N)
    y = np.linspace(a, b, N)
    X, Y = np.meshgrid(x, y)
    x = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32) #shape = (N*N, 2)
    x = torch.cat([x, torch.zeros(x.shape[0], u.dim-2)], dim=1)  #shape = (N*N, 10)
    x = x.unsqueeze(0).repeat(bs, 1, 1).to(device)  #shape = (bs, N*N, 10)
    
    #--- node ----
    v = V(dim=u.n_params, dropout_p=dropout_p).to(device)
    v = v.to(device)
    if test_num:
        ckp_path = f"checkpoints/{eqn_config.eqn_type}_dim{u.dim}_drop1_cso_c{c_type}_n{test_num}.pth"
    else:
        ckp_path = f"checkpoints/{eqn_config.eqn_type}_dim{u.dim}_drop1_cso_c{c_type}.pth"
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    v.load_state_dict(checkpoint['v_model_state_dict'])
    with torch.no_grad():
        thetaT_pred, node_pred_res = inference_2(theta_0, u, v, None, x, eqn, False, c_type)
        node_pred_sol = u.forward_2(thetaT_pred, x)     #shape = (bs, N*N)

    #--- pdqp ----
    v = V(dim=u.n_params, dropout_p=dropout_p).to(device)
    v = v.to(device)

    # pdqp net
    feat_sizes = [eqn_config.feat_size] * eqn_config.feat_depth
    pdqp = Adaptive_PDQP_Net(feat_sizes, dropout_p)
    pdqp = pdqp.to(device)

    if test_num:
        ckp_path = f"checkpoints/{eqn_config.eqn_type}_dim{u.dim}_drop1_heap_c{c_type}_n{test_num}.pth"
    else:
        ckp_path = f"checkpoints/{eqn_config.eqn_type}_dim{u.dim}_drop1_heap_c{c_type}.pth"
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    v.load_state_dict(checkpoint['v_model_state_dict'])
    pdqp.load_state_dict(checkpoint['pdqp_model_state_dict'])
    with torch.no_grad():
        thetaT_pred, pdqp_pred_res = inference_2(theta_0, u, v, pdqp, x, eqn, True, c_type) # need retest
        pdqp_pred_sol = u.forward_2(thetaT_pred, x)     #shape = (bs, N*N)

    #   draw bs subplots, each subplot contains two lines: label and pred
    # 计算带 res 的预测结果的最小值和最大值
    node_pred_res = node_pred_res * rate
    pdqp_pred_res = pdqp_pred_res * rate
    all_pred_res_values = np.concatenate([node_pred_res[i].cpu().numpy().flatten() for i in range(bs)] + 
                                        [pdqp_pred_res[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_res = all_pred_res_values.min()
    vmax_res = all_pred_res_values.max()

    levels_res = np.linspace(vmin_res, vmax_res, 100)

    # 计算不带 res 的预测结果的最小值和最大值
    all_pred_val_values = np.concatenate([node_pred_sol[i].cpu().numpy().flatten() for i in range(bs)] + 
                                            [pdqp_pred_sol[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_val = all_pred_val_values.min()
    vmax_val = all_pred_val_values.max()

    levels_val = np.linspace(vmin_val, vmax_val, 100)

    colormap = plt.cm.viridis # inferno, plasma

    # 生成子图
    plt.rcParams['axes.titlesize'] = 25  # 设置默认标题大小
    fig, axs = plt.subplots(nrows=bs*2, ncols=2, figsize=(14, 14), constrained_layout=True)

    for i in range(bs):
        print(node_pred_res[i].mean(), pdqp_pred_res[i].mean())

        # 不带 res 的子图
        ax = axs[i*2, 0]
        pred_z = node_pred_sol[i].cpu().numpy().reshape(X.shape)
        ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_val)
        ax.set_title(f"Pred_u (CSO)", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        ax = axs[i*2, 1]
        pred_z = pdqp_pred_sol[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_val)
        ax.set_title(f"Pred_u (HEAP)", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2, :])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

        # 带 res 的子图
        ax = axs[i*2+1, 0]
        pred_z = node_pred_res[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_res)
        ax.set_title(f"Res (CSO)", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        ax = axs[i*2+1, 1]
        pred_z = pdqp_pred_res[i].cpu().numpy().reshape(X.shape)
        ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_res)
        ax.set_title(f"Res (HEAP)", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2+1, :])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

    plt.savefig(f"pics/{eqn_config.eqn_type}_dim{u.dim}_c{c_type}.png", dpi=200)

def visualize_pricing_check(eqn_config_1, eqn_config_2, rand_num):
    #--- load data ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u_1 = U_TWO(eqn_config_1.dim, hidden_dim1=40, hidden_dim2=40).to(device)
    u_2 = U_TWO(eqn_config_2.dim, hidden_dim1=40, hidden_dim2=40).to(device)
    dataset_test = torch.load(f'checkpoints/pricing_default_risk_dataset_dim{u_1.dim}_params{u_1.n_params}_seed{eqn_config_1.seed}_rand.pt', weights_only=False)
    
    bs = 2
    theta_0 = dataset_test.tensors[0][rand_num*bs:(rand_num+1)*bs].to(device)
    mask = [True, False]
    theta_0 = theta_0[mask]
    bs = len(theta_0)
    T = eqn_config_1.total_time
    n_t = eqn_config_1.num_time_interval
    if eqn_type == "pricing_default_risk":
        eqn_1 = PricingDefaultRisk(eqn_config_1)
        eqn_2 = PricingDefaultRisk(eqn_config_2)
    else:
        eqn_1 = None
        eqn_2 = None

    #  --- define x, only first two dimensions non-zero ---- 
    N = 50
    a = 0.0 #left boundary
    b = 1.0 #right boundary
    x = np.linspace(a, b, N)
    y = np.linspace(a, b, N)
    X, Y = np.meshgrid(x, y)
    x = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32) #shape = (N*N, 2)
    x = torch.cat([x, torch.zeros(x.shape[0], u_1.dim-2)], dim=1)  #shape = (N*N, 10)
    x = x.unsqueeze(0).repeat(bs, 1, 1).to(device)  #shape = (bs, N*N, 10)
    
    #--- pdqp ----
    v_1 = V(dim=u_1.n_params, dropout_p=eqn_config_1.train_dropout).to(device)
    v_1 = v_1.to(device)

    v_2 = V(dim=u_2.n_params, dropout_p=eqn_config_2.train_dropout).to(device)
    v_2 = v_2.to(device)

    # pdqp net
    feat_sizes = [eqn_config_1.feat_size] * eqn_config_1.feat_depth
    pdqp_1 = Adaptive_PDQP_Net(feat_sizes, eqn_config_1.train_dropout)
    pdqp_1 = pdqp_1.to(device)

    feat_sizes = [eqn_config_2.feat_size] * eqn_config_2.feat_depth
    pdqp_2 = Adaptive_PDQP_Net(feat_sizes, eqn_config_2.train_dropout)
    pdqp_2 = pdqp_2.to(device)

    ckp_path = f"checkpoints/pricing_default_risk_dim{u_1.dim}_drop1_pdqp_n1.pth"
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    v_1.load_state_dict(checkpoint['v_model_state_dict'])
    pdqp_1.load_state_dict(checkpoint['pdqp_model_state_dict'])
    with torch.no_grad():
        thetaT_pred, node_pred_res = inference_2(theta_0, u_1, v_1, pdqp_1, x, eqn_1, True) # need retest
        # thetaT_pred, _ = inference_2(theta_0, u_1, v_1, pdqp_1, x, eqn_1, False) # need retest
        node_pred_sol = u_1.forward_2(thetaT_pred, x)     #shape = (bs, N*N)


    ckp_path = f"checkpoints/pricing_default_risk_dim{u_2.dim}_drop1_pdqp_n2.pth"
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    v_2.load_state_dict(checkpoint['v_model_state_dict'])
    pdqp_2.load_state_dict(checkpoint['pdqp_model_state_dict'])
    with torch.no_grad():
        thetaT_pred, pdqp_pred_res = inference_2(theta_0, u_2, v_2, pdqp_2, x, eqn_2, True) # need retest
        pdqp_pred_sol = u_2.forward_2(thetaT_pred, x)     #shape = (bs, N*N)

    diff_pred_sol = torch.abs(pdqp_pred_sol - node_pred_sol)
    diff_pred_res = torch.abs(pdqp_pred_res - node_pred_res)

    #   draw bs subplots, each subplot contains two lines: label and pred
    # 计算带 res 的预测结果的最小值和最大值
    all_pred_res_values = np.concatenate([node_pred_res[i].cpu().numpy().flatten() for i in range(bs)] + 
                                        [pdqp_pred_res[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_res = all_pred_res_values.min()
    vmax_res = all_pred_res_values.max()

    levels_res = np.linspace(vmin_res, vmax_res, 100)

    # 计算不带 res 的预测结果的最小值和最大值
    all_pred_val_values = np.concatenate([node_pred_sol[i].cpu().numpy().flatten() for i in range(bs)] + 
                                            [pdqp_pred_sol[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_val = all_pred_val_values.min()
    vmax_val = all_pred_val_values.max()

    levels_val = np.linspace(vmin_val, vmax_val, 100)

    all_diff_res_values = np.concatenate([diff_pred_res[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_res = all_diff_res_values.min()
    vmax_res = all_diff_res_values.max()

    levels_diff_res = np.linspace(vmin_res, vmax_res, 100)

    all_diff_val_values = np.concatenate([diff_pred_sol[i].cpu().numpy().flatten() for i in range(bs)])
    vmin_val = all_diff_val_values.min()
    vmax_val = all_diff_val_values.max()

    levels_diff_val = np.linspace(vmin_val, vmax_val, 100)

    colormap = plt.cm.viridis # inferno, plasma
    colormap_s = plt.cm.inferno

    # 生成子图
    plt.rcParams['axes.titlesize'] = 25  # 设置默认标题大小
    fig, axs = plt.subplots(nrows=bs*2, ncols=3, figsize=(25, 16), constrained_layout=True)

    for i in range(bs):
        print(node_pred_res[i].mean(), pdqp_pred_res[i].mean())

        # 不带 res 的子图
        ax = axs[i*2, 0]
        pred_z = node_pred_sol[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_val)
        ax.set_title(f"Pred_u_c1", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        ax = axs[i*2, 1]
        pred_z = pdqp_pred_sol[i].cpu().numpy().reshape(X.shape)
        ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_val)
        ax.set_title(f"Pred_u_c2", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2, 1])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

        ax = axs[i*2, 2]
        pred_z = diff_pred_sol[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap_s, levels=levels_diff_val)
        ax.set_title(f"Pred_u_diff", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2, 2])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

        # 带 res 的子图
        ax = axs[i*2+1, 0]
        pred_z = node_pred_res[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_res)
        ax.set_title(f"Res_c1", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        ax = axs[i*2+1, 1]
        pred_z = pdqp_pred_res[i].cpu().numpy().reshape(X.shape)
        ax.contourf(X, Y, pred_z, cmap=colormap, levels=levels_res)
        ax.set_title(f"Res_c2", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2+1, 1])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

        ax = axs[i*2+1, 2]
        pred_z = diff_pred_res[i].cpu().numpy().reshape(X.shape)
        cs_ref = ax.contourf(X, Y, pred_z, cmap=colormap_s, levels=levels_diff_res)
        ax.set_title(f"Res_diff", fontsize=35, fontweight='bold')  # 加大加粗标题
        ax.set_xticks([a, b])
        ax.set_yticks([a, b])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=20, width=2)  # 设置坐标轴刻度字体大小和加粗

        # 设置颜色条并格式化
        cbar = fig.colorbar(cs_ref, ax=axs[i*2+1, 2])
        cbar.ax.tick_params(labelsize=15)  # 设置颜色条数值的字体大小
        # cbar.set_ticks([round(i, 2) for i in cbar.get_ticks()])  # 格式化为两位小数

    plt.savefig(f"pics/pricing_pred_res_{u_1.dim}_{rand_num}.png", dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="方程类型，维度以及是否dropout")
    parser.add_argument('--eqn_type', type=str, default='pricing_default_risk', help='方程类型')
    parser.add_argument('--cur_dim', type=int, default=10, help='数据维度')
    parser.add_argument('--rand_num', type=int, default=4, help='样本随机数')
    parser.add_argument('--test_num', type=int, default=0, help='测试分支')


    args = parser.parse_args()
    eqn_type = args.eqn_type
    cur_dim = args.cur_dim
    rand_num = args.rand_num
    test_num = args.test_num

    eqn_config = EqnConfig(eqn_type=eqn_type, dim=cur_dim, num=test_num)

    
    # eqn_config_1 = EqnConfig(eqn_type=eqn_type, dim=cur_dim, num=1)
    # eqn_config_2 = EqnConfig(eqn_type=eqn_type, dim=cur_dim, num=2)
    set_gpu_max_mem()
    visualize_pricing(eqn_config, test_num, rand_num)
    # visualize_pricing_check(eqn_config_1, eqn_config_2, rand_num)