import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

model_name = "llama3"


# 加载数据集
def load_and_prepare_data(file_path):
    data = Dataset.from_json(file_path)
    return data.train_test_split(test_size=0.1)


# 数据预处理
# def preprocess_function(examples):
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     return tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# 数据预处理
def preprocess_function(examples):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=512)

    # 将输出的input_ids移位，以创建目标标签
    outputs['labels'] = outputs['input_ids'].copy()

    return {**inputs, **outputs}


# 加载和准备数据
file_path = 'training_jsons/ai_customer_service_QA1.jsonl'
dataset = load_and_prepare_data(file_path)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 加载模型
model = LlamaForSequenceClassification.from_pretrained(model_name)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# 训练模型
trainer.train()

# 保存模型
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
