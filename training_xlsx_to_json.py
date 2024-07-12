import pandas as pd
import json

def xlsx_to_json(input_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 构造JSON数组
    json_array = []
    for index, row in df.iterrows():
        # 构造JSON对象
        json_object = {"input": row["Q"], "output": row["A"]}
        # 添加到数组
        json_array.append(json_object)

    # 将JSON数组写入文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_array, json_file, ensure_ascii=False, indent=2)

# 使用示例
xlsx_to_json('training_xlsx/chainup_project_info.xlsx', 'training_jsons/chainup_project_info.json')
