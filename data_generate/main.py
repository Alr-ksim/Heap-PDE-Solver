import json
import munch
import os
import logging
from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

import utils
import equation as eqn
from solver import BSDESolver

flags.DEFINE_string('exp_name', 'hjb_lq',
                    """The name of numerical experiments, prefix for logging""")
flags.DEFINE_integer('data_dim', 5, """The dim of the x in PDE""")
flags.DEFINE_integer('n', 1, """The experiment index for multiple runs""")  # 定义实验编号
flags.DEFINE_integer('num_threads', 4, """Number of threads for parallel execution.""")  # 定义线程数量
FLAGS = flags.FLAGS
FLAGS.log_dir = './data_generate/logs'  # 日志和输出的保存路径

def run_experiment(n, exp_name, data_dim):
    """运行单个实验的函数"""
    # 根据方程类型选择合适的配置生成函数
    if "burgers_type" in exp_name:
        generate_func = utils.generate_burgers_config
    elif "pricing_default_risk" in exp_name:
        generate_func = utils.generate_pricing_default_risk_config
    elif "pricing_diffrate" in exp_name:
        generate_func = utils.generate_pricing_diffrate_config
    elif "reaction_diffusion" in exp_name:
        generate_func = utils.generate_reaction_diffusion_config
    elif "hjb_lq" in exp_name:
        generate_func = utils.generate_HJBLQ_config
    elif "allencahn" in exp_name:
        generate_func = utils.generate_allencahn_config
    else:
        raise ValueError("Unsupported equation type.")

    
    config_path = f"data_generate/raw_configs/{exp_name}_d{data_dim}.json"
    # 生成新的配置文件名
    config_output_path = f'data_generate/configs/{exp_name}_d{data_dim}_{n}.json'
    
    # 生成新配置文件，调整 g_param 的范围
    generate_func(config_path, config_output_path)

    # 使用生成的配置文件进行求解
    with open(config_output_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    # 创建日志文件夹
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()

    # 记录求解结果
    logging.info('Begin recording solution...')
    final_loss = training_history[-1, 1]  # 获取最后一轮的loss
    records = bsde_solver.record_solution(batch_size=256, final_loss=final_loss)

    # 确保数据文件夹存在
    data_path = f'./data_generate/data_{exp_name}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # 保存记录的数据到 /data 文件夹
    records_output_path = os.path.join(data_path, f'{exp_name}_d{data_dim}_{n}_records.json')
    with open(records_output_path, 'w') as f:
        json.dump(records, f, indent=4)

    logging.info('Recorded solution saved to: %s', records_output_path)

    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init))


def main(argv):
    del argv
    exp_name = FLAGS.exp_name
    data_dim = FLAGS.data_dim
    start_n = FLAGS.n  # 保存初始的 n 值

    # 创建一个线程池，运行多个实验
    with ThreadPoolExecutor(max_workers=FLAGS.num_threads) as executor:
        futures = [executor.submit(run_experiment, start_n + i, exp_name, data_dim) for i in range(FLAGS.num_threads)]
        for future in futures:
            future.result()  # 等待所有线程完成

if __name__ == '__main__':
    app.run(main)
