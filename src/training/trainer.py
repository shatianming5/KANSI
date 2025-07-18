import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
import numpy as np
from utils.device_utils import DeviceManager
from utils.config import TRAINING_OPTIMIZATION
from utils.monitor import TrainingMonitor, EarlyStoppingMonitor

class CustomTrainingCallback(TrainerCallback):
    """自定义训练回调"""
    
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
        self.epoch_start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """轮次开始"""
        self.monitor.start_epoch(state.epoch)
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """轮次结束"""
        # 从日志历史中获取最新的训练损失
        train_loss = state.log_history[-1].get('train_loss', 0) if state.log_history else 0
        
        # 获取评估指标
        eval_metrics = {}
        for log in reversed(state.log_history):
            if 'eval_loss' in log:
                eval_metrics = {k: v for k, v in log.items() if k.startswith('eval_')}
                break
                
        self.monitor.end_epoch(state.epoch, train_loss, eval_metrics)
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始"""
        config = {
            "model_name": args.output_dir,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate
        }
        self.monitor.start_training(int(args.num_train_epochs), config)
        
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束"""
        self.monitor.end_training()

class ModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device_manager = DeviceManager()
        
        # 将模型移动到GPU
        self.model = self.device_manager.move_to_device(self.model)
        
        # 打印设备信息
        self.device_manager.print_device_info()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化监控器
        self.monitor = TrainingMonitor("./logs")
        self.early_stopping_monitor = None
        
        # 训练历史
        self.training_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": [],
            "epochs": []
        }
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def setup_training_args(self):
        """设置训练参数"""
        training_args = {
            "output_dir": self.config["output_dir"],
            "num_train_epochs": self.config["num_epochs"],
            "per_device_train_batch_size": self.config["batch_size"],
            "per_device_eval_batch_size": self.config["batch_size"] * 2,
            "eval_strategy": "epoch",
            "logging_steps": 50,
            "save_strategy": "epoch",
            "load_best_model_at_end": TRAINING_OPTIMIZATION["load_best_model_at_end"],
            "metric_for_best_model": TRAINING_OPTIMIZATION["metric_for_best_model"],
            "greater_is_better": TRAINING_OPTIMIZATION["greater_is_better"],
            "learning_rate": self.config["learning_rate"],
            "warmup_ratio": TRAINING_OPTIMIZATION["warmup_ratio"],
            "weight_decay": TRAINING_OPTIMIZATION["weight_decay"],
            "logging_dir": "./logs",
            "report_to": "none",
            "save_total_limit": 3,
            "seed": 42,
            "fp16": False,
            "dataloader_num_workers": 0,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": TRAINING_OPTIMIZATION["gradient_clipping"],
            "lr_scheduler_type": TRAINING_OPTIMIZATION["learning_rate_scheduler"],
            "logging_first_step": True,
            "disable_tqdm": False
        }
        
        # GPU特定优化
        if self.device_manager.device.type == "cuda":
            training_args["fp16"] = True
            training_args["dataloader_num_workers"] = 4
            training_args["gradient_accumulation_steps"] = 2
            training_args["per_device_train_batch_size"] = min(16, self.config["batch_size"] * 2)
            self.logger.info(f"启用GPU加速训练，优化批次大小: {training_args['per_device_train_batch_size']}")
        elif self.device_manager.device.type == "mps":
            training_args["fp16"] = False
            training_args["dataloader_num_workers"] = 2
            self.logger.info(f"启用Apple Silicon GPU训练，共{self.config['num_epochs']}轮")
        else:
            training_args["dataloader_num_workers"] = 2
            self.logger.info(f"使用CPU训练，共{self.config['num_epochs']}轮")
        
        return TrainingArguments(**training_args)
    
    def setup_callbacks(self):
        """设置回调函数"""
        callbacks = []
        
        # 添加自定义监控回调
        callbacks.append(CustomTrainingCallback(self.monitor))
        
        # 早停机制
        if TRAINING_OPTIMIZATION["use_early_stopping"]:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=TRAINING_OPTIMIZATION["early_stopping_patience"],
                early_stopping_threshold=TRAINING_OPTIMIZATION["early_stopping_threshold"]
            )
            callbacks.append(early_stopping)
            
            # 初始化自定义早停监控
            self.early_stopping_monitor = EarlyStoppingMonitor(
                patience=TRAINING_OPTIMIZATION["early_stopping_patience"],
                min_delta=TRAINING_OPTIMIZATION["early_stopping_threshold"],
                mode="max"
            )
            
            self.logger.info(f"启用早停机制，耐心值: {TRAINING_OPTIMIZATION['early_stopping_patience']}")
        
        return callbacks
    
    def train(self, train_dataset, eval_dataset):
        """训练模型"""
        self.logger.info("="*60)
        self.logger.info(f"开始训练 - 模式: {self.config.get('training_mode', 'standard')}")
        self.logger.info(f"训练轮数: {self.config['num_epochs']}")
        self.logger.info(f"训练数据量: {len(train_dataset)}")
        self.logger.info(f"验证数据量: {len(eval_dataset)}")
        self.logger.info(f"批次大小: {self.config['batch_size']}")
        self.logger.info(f"学习率: {self.config['learning_rate']}")
        self.logger.info(f"设备: {self.device_manager.device}")
        self.logger.info("="*60)
        
        training_args = self.setup_training_args()
        callbacks = self.setup_callbacks()
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # 训练开始时间
        start_time = time.time()
        
        # 清理缓存
        self.device_manager.clear_cache()
        
        try:
            # 开始训练
            train_result = trainer.train()
            
            # 训练结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            # 最终评估
            eval_result = trainer.evaluate()
            
            # 获取训练进度信息
            progress = self.monitor.get_training_progress()
            performance = self.monitor.get_performance_summary()
            
            # 输出训练结果
            self.logger.info("="*60)
            self.logger.info("训练完成摘要:")
            self.logger.info(f"  实际训练时间: {training_time:.2f} 秒")
            self.logger.info(f"  完成轮数: {progress['current_epoch']}/{progress['total_epochs']}")
            self.logger.info(f"  最佳F1分数: {performance['最佳F1分数']:.4f} (第{performance['最佳轮次']}轮)")
            self.logger.info(f"  最终准确率: {eval_result.get('eval_accuracy', 0):.4f}")
            self.logger.info(f"  最终F1分数: {eval_result.get('eval_f1', 0):.4f}")
            self.logger.info(f"  平均训练速度: {performance['平均训练速度']:.2f} 秒/轮")
            
            if performance['是否早停']:
                self.logger.info(f"  训练状态: 早停 (节省了 {self.config['num_epochs'] - progress['current_epoch']} 轮)")
            else:
                self.logger.info(f"  训练状态: 正常完成")
                
            self.logger.info("="*60)
            
            # 保存模型
            if TRAINING_OPTIMIZATION["save_best_model"]:
                self.model.save_pretrained(self.config["output_dir"])
                self.tokenizer.save_pretrained(self.config["output_dir"])
                self.logger.info(f"最佳模型已保存到: {self.config['output_dir']}")
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
        finally:
            # 清理缓存
            self.device_manager.clear_cache()
        
        return trainer, {
            "训练时间": training_time,
            "训练损失": train_result.training_loss,
            "验证结果": eval_result,
            "训练轮数": progress['current_epoch'],
            "最佳轮数": performance['最佳轮次'],
            "最佳F1分数": performance['最佳F1分数'],
            "是否早停": performance['是否早停'],
            "训练进度": progress,
            "性能摘要": performance
        }
    
    def evaluate(self, eval_dataset):
        """单独评估模型性能"""
        self.logger.info("开始评估模型")
        
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        eval_results = trainer.evaluate()
        
        self.logger.info(f"评估结果: {eval_results}")
        
        return eval_results
    
    def get_training_summary(self):
        """获取训练摘要"""
        return {
            "训练配置": {
                "训练轮数": self.config["num_epochs"],
                "批次大小": self.config["batch_size"],
                "学习率": self.config["learning_rate"],
                "设备": str(self.device_manager.device)
            },
            "优化配置": TRAINING_OPTIMIZATION,
            "训练历史": self.training_history,
            "监控数据": self.monitor.get_performance_summary()
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        try:
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
