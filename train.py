import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 参数
model_path = "bert-base-chinese"
max_len, bs, epochs = 128, 8, 3
out_dir = "./ccpoem_classifier"

# 加载数据
df_train = pd.read_csv("train_all.csv")[["text","label"]]
df_val = pd.read_csv("val_all.csv")[["text","label"]]
train_ds = Dataset.from_pandas(df_train)
val_ds = Dataset.from_pandas(df_val)

# Tokenizer / 模型加载
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=max_len)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

# 定义评估
def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy_score(labels, preds), "f1": f1, "precision": p, "recall": r}

# 训练配置
args = TrainingArguments(
    output_dir=out_dir, num_train_epochs=epochs,
    per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    eval_strategy="epoch", logging_steps=50,
    save_strategy="epoch", load_best_model_at_end=True
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  tokenizer=tokenizer, compute_metrics=compute_metrics)

# 开始训练
trainer.train()
model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
print("✅ 微调训练完成，模型保存在", out_dir)
