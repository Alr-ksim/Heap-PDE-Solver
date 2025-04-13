import os
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_record(record):
    processed_data = []
    for t, x_list, u_list in record["data"]:
        if not t:
            processed_data.append([t, x_list[0], u_list[0]])
        else:
            for x, u in zip(x_list, u_list):
                processed_data.append([t, x, u])
    return {
        "g_param": record["g_param"],
        "final_loss": record["final_loss"],
        "data": processed_data
    }

def main():
    exp_name = "reaction_diffusion"
    data_dim = 5

    data_folder = f'./data_generate/data_{exp_name}'
    keys = f'{exp_name}_d{data_dim}_'
    record_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith(keys) and f.endswith('_records.json')]

    # 加载所有记录，并按 loss 排序
    records = []
    for file_path in record_files:
        record = load_data(file_path)
        records.append(record)
    
    # 按 loss 排序，选择最小的 128 条
    records_sorted = sorted(records, key=lambda x: x['final_loss'])
    top_200_records = records_sorted[:128]

    # 处理并整合数据
    processed_records = [process_record(record) for record in top_200_records]

    # 保存整合后的数据到新的 json 文件
    output_file = f'./checkpoints/{keys}combined_records.json'
    with open(output_file, 'w') as outfile:
        json.dump(processed_records, outfile, indent=4)

    print(f"Combined records saved to {output_file}")

if __name__ == "__main__":
    main()
