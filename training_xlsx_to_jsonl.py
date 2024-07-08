import pandas as pd
import json


def xlsx_to_jsonl(input_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        # 遍历每一行
        for index, row in df.iterrows():
            # 构造JSON对象
            json_object = {"input": row["Q"], "output": row["A"]}
            # 将JSON对象写入文件
            jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')


# 使用示例
xlsx_to_jsonl('training_xlsx/ai_customer_service_QA1.xlsx', 'training_jsons/ai_customer_service_QA1.jsonl')
