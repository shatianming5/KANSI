import os
import torch

# 环境配置
HF_ENDPOINT = "https://hf-mirror.com"
os.environ['HF_ENDPOINT'] = HF_ENDPOINT

# 设备检测函数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# 训练轮数配置
TRAINING_EPOCHS = {
    "quick": 3,
    "standard": 5,
    "intensive": 8,
    "maximum": 10
}

# 模型配置
MODEL_CONFIG = {
    "model_name": "bert-base-chinese",
    "num_labels": 2,
    "max_length": 128,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 8,
    "output_dir": "./models/ccpoem_classifier",
    "device": get_device(),
    "training_mode": "intensive"
}

# 训练参数优化配置
TRAINING_OPTIMIZATION = {
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    "learning_rate_scheduler": "linear",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    "save_best_model": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True
}

# 数据路径配置
DATA_CONFIG = {
    "train_file": "train_all.csv",
    "val_file": "val_all.csv",
    "text_column": "text",
    "label_column": "label"
}

# 更新训练模式函数
def update_training_mode(mode):
    if mode not in TRAINING_EPOCHS:
        raise ValueError(f"训练模式必须是: {list(TRAINING_EPOCHS.keys())}")
    
    MODEL_CONFIG["training_mode"] = mode
    MODEL_CONFIG["num_epochs"] = TRAINING_EPOCHS[mode]
    
    print(f"训练模式已更新为: {mode} ({TRAINING_EPOCHS[mode]} 轮)")
    return TRAINING_EPOCHS[mode]

def get_training_config():
    return {
        "当前训练模式": MODEL_CONFIG["training_mode"],
        "训练轮数": MODEL_CONFIG["num_epochs"],
        "学习率": MODEL_CONFIG["learning_rate"],
        "批次大小": MODEL_CONFIG["batch_size"],
        "早停机制": TRAINING_OPTIMIZATION["use_early_stopping"],
        "早停耐心值": TRAINING_OPTIMIZATION["early_stopping_patience"],
        "设备": str(MODEL_CONFIG["device"])
    }

# 测试代码
if __name__ == "__main__":
    print("=== 配置测试 ===")
    for key, value in MODEL_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n=== 训练配置 ===")
    config = get_training_config()
    for key, value in config.items():
        print(f"{key}: {value}")
