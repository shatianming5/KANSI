import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# 路径配置
input_json = "data/japanpoetry_sentences.json"
output_pt = "embeddings/japan_sentences_embeddings_meanpool.pt"

# 加载 BERT 模型
tokenizer = AutoTokenizer.from_pretrained("./models/ccpoem_classifier")
model = AutoModel.from_pretrained("./models/ccpoem_classifier")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载数据
with open(input_json, "r", encoding="utf-8") as f:
    sentences = json.load(f)

# 存储结构
ids = []
texts = []
embeddings = []

for item in tqdm(sentences):
    pid = item["id"]
    text = item["sentence"].strip()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state            # shape: [1, seq_len, 768]
        attention_mask = inputs["attention_mask"]          # shape: [1, seq_len]

        # Mean Pooling：用 attention mask 来屏蔽 padding 区域
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()  # [1, seq_len, 768]
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts

    ids.append(pid)
    texts.append(text)
    embeddings.append(mean_pooled.squeeze(0).cpu())  # shape: [768]

# 打包保存为 .pt 文件
embeddings_tensor = torch.stack(embeddings)

torch.save({
    "ids": ids,
    "texts": texts,
    "embeddings": embeddings_tensor
}, output_pt)

print(f"✅ 已保存 mean pooling 嵌入文件：{output_pt}，共 {len(ids)} 条")
