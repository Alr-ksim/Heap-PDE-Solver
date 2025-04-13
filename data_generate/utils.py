import json
import numpy as np

def generate_reaction_diffusion_config(base_config_path, output_config_path):
    """
    生成ReactionDiffusionType类的配置文件，动态调整g_param和y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = config['net_config']['y_init_range'][0] * g_param
    y_init_upper = config['net_config']['y_init_range'][1] * g_param
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated config with g_param={g_param}, saved to {output_config_path}")

def generate_burgers_config(base_config_path, output_config_path):
    """
    生成BurgersType类的配置文件，动态调整g_param和y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = 1 - 1.0 / (1 + np.exp(0 + g_param / config["eqn_config"]["dim"]))
    y_init_upper = 1 - 1.0 / (1 + np.exp(0.3 + g_param / config["eqn_config"]["dim"]))
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated config with g_param={g_param}, saved to {output_config_path}")

def generate_pricing_default_risk_config(base_config_path, output_config_path):
    """
    生成 PricingDefaultRisk 类的配置文件，动态调整 g_param 和 y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param 的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成 g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = config["net_config"]["y_init_range"][0] * g_param  # 假设与 g_param 有线性关系
    y_init_upper = config["net_config"]["y_init_range"][1] * g_param
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated PricingDefaultRisk config with g_param={g_param}, saved to {output_config_path}")

def generate_pricing_diffrate_config(base_config_path, output_config_path):

    """
    生成 PricingDiffRate 类的配置文件，动态调整 g_param 和 y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param 的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成 g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = config["net_config"]["y_init_range"][0]
    y_init_upper = config["net_config"]["y_init_range"][1]
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated PricingDiffRate config with g_param={g_param}, saved to {output_config_path}")

def generate_HJBLQ_config(base_config_path, output_config_path):

    """
    生成HJBLQ类的配置文件，动态调整g_param和y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = config['net_config']['y_init_range'][0] * g_param
    y_init_upper = config['net_config']['y_init_range'][1] * g_param
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated config with g_param={g_param}, saved to {output_config_path}")

def generate_allencahn_config(base_config_path, output_config_path):

    """
    生成AllenCahn类的配置文件，动态调整g_param和y_init_range。
    :param base_config_path: 基本配置文件路径
    :param output_config_path: 输出配置文件路径
    :param g_param_range: g_param的范围 (min_value, max_value)
    """
    # 读取基本配置文件
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # 在合理范围内生成g_param
    g_param_range = [config["eqn_config"]["g_l"], config["eqn_config"]["g_r"]]
    g_param = np.random.uniform(g_param_range[0], g_param_range[1])
    del config["eqn_config"]["g_l"]
    del config["eqn_config"]["g_r"]

    # 动态调整 y_init_range，假设它与 g_param 相关
    y_init_lower = config['net_config']['y_init_range'][0] * g_param
    y_init_upper = config['net_config']['y_init_range'][1] * g_param
    config["net_config"]["y_init_range"] = [y_init_lower, y_init_upper]

    # 将 g_param 作为一个额外配置项加入配置文件
    config["eqn_config"]["g_param"] = g_param

    # 保存新配置文件
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated config with g_param={g_param}, saved to {output_config_path}")