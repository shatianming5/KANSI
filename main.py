import sys
import os
import argparse
sys.path.append('src')

from utils.config import MODEL_CONFIG, DATA_CONFIG, update_training_mode, get_training_config
from utils.device_utils import DeviceManager
from data_processing.data_loader import DataLoader
from data_processing.preprocessor import TextPreprocessor
from model.bert_classifier import BertClassifier
from training.trainer import ModelTrainer
from utils.evaluator import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="唐诗元白风格分类模型训练")
    parser.add_argument('--mode', type=str, default='intensive', 
                       choices=['quick', 'standard', 'intensive', 'maximum'],
                       help='训练模式: quick(3轮), standard(5轮), intensive(8轮), maximum(10轮)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='训练样本数量（用于测试）')
    parser.add_argument('--full_dataset', action='store_true',
                       help='使用完整数据集训练')
    
    args = parser.parse_args()
    
    print("开始训练唐诗元白风格分类模型...")
    
    # 设置训练模式
    update_training_mode(args.mode)
    
    # 显示当前配置
    print("\n当前训练配置:")
    config = get_training_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 0. 显示设备信息
    print("设备信息:")
    device_manager = DeviceManager()
    device_manager.print_device_info()
    print()
    
    # 1. 加载数据
    print("1. 加载数据...")
    data_loader = DataLoader(DATA_CONFIG)
    
    if args.full_dataset:
        df_train, df_val = data_loader.load_data(
            DATA_CONFIG["train_file"], 
            DATA_CONFIG["val_file"]
        )
        print("使用完整数据集训练")
    else:
        df_train, df_val = data_loader.load_data(
            DATA_CONFIG["train_file"], 
            DATA_CONFIG["val_file"],
            sample_size=args.sample_size
        )
        print(f"使用样本数据集训练，样本数量: {args.sample_size}")
    
    print(f"训练集样本数: {len(df_train)}")
    print(f"验证集样本数: {len(df_val)}")
    
    # 2. 创建数据集
    print("2. 创建数据集...")
    train_ds, val_ds = data_loader.create_datasets(df_train, df_val)
    
    # 3. 预处理数据
    print("3. 预处理数据...")
    preprocessor = TextPreprocessor(MODEL_CONFIG["model_name"], MODEL_CONFIG["max_length"])
    train_ds, val_ds = preprocessor.preprocess_datasets(train_ds, val_ds)
    
    # 4. 加载模型
    print("4. 加载模型...")
    classifier = BertClassifier(MODEL_CONFIG["model_name"], MODEL_CONFIG["num_labels"])
    model, tokenizer = classifier.load_model()
    
    # 显示模型信息
    model_info = classifier.get_model_info()
    print(f"模型参数量: {model_info['总参数量']:,}")
    print(f"可训练参数量: {model_info['可训练参数量']:,}")
    print(f"模型设备: {model_info['模型设备']}")
    print()
    
    # 5. 训练模型
    print("5. 开始训练...")
    os.makedirs(MODEL_CONFIG["output_dir"], exist_ok=True)
    trainer = ModelTrainer(model, tokenizer, MODEL_CONFIG)
    trainer_obj, training_results = trainer.train(train_ds, val_ds)
    
    # 显示训练结果
    print("\n训练结果摘要:")
    for key, value in training_results.items():
        if key != "验证结果":
            print(f"  {key}: {value}")
    
    if "验证结果" in training_results:
        print("  验证结果:")
        for key, value in training_results["验证结果"].items():
            if key.startswith("eval_"):
                print(f"    {key}: {value:.4f}")
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    trained_classifier = BertClassifier(MODEL_CONFIG["model_name"], MODEL_CONFIG["num_labels"])
    trained_classifier.load_trained_model(MODEL_CONFIG["output_dir"])
    
    evaluator = ModelEvaluator(trained_classifier)
    results = evaluator.evaluate_on_dataset(df_val, sample_size=50)
    
    print(f"\n最终评估结果:")
    print(f"  验证集准确率: {results['accuracy']:.2%}")
    print(f"  训练模式: {args.mode} ({MODEL_CONFIG['num_epochs']}轮)")
    print(f"  训练时间: {training_results['训练时间']:.2f}秒")
    print("\n✅ 训练完成！")

if __name__ == "__main__":
    main()
