import time
import json
import os
from typing import Dict, List, Any
import logging
from datetime import datetime

class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, output_dir: str = "./logs"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_monitor.log")
        self.metrics_file = os.path.join(output_dir, "training_metrics.json")
        
        # 创建日志目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 监控数据
        self.training_metrics = {
            "start_time": None,
            "end_time": None,
            "total_epochs": 0,
            "current_epoch": 0,
            "train_losses": [],
            "eval_losses": [],
            "eval_accuracies": [],
            "eval_f1_scores": [],
            "learning_rates": [],
            "training_speed": [],
            "best_epoch": 0,
            "best_f1": 0.0,
            "early_stopped": False,
            "early_stopping_epoch": None
        }
        
        # 性能监控
        self.epoch_start_time = None
        self.last_log_time = time.time()
        
    def start_training(self, total_epochs: int, config: Dict[str, Any]):
        """开始训练监控"""
        self.training_metrics["start_time"] = datetime.now().isoformat()
        self.training_metrics["total_epochs"] = total_epochs
        self.training_metrics["config"] = config
        
        self.logger.info(f"开始训练监控 - 总轮数: {total_epochs}")
        self.logger.info(f"训练配置: {config}")
        
    def start_epoch(self, epoch: int):
        """开始新轮次"""
        self.training_metrics["current_epoch"] = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f"开始第 {epoch + 1}/{self.training_metrics['total_epochs']} 轮训练")
        
    def end_epoch(self, epoch: int, train_loss: float, eval_metrics: Dict[str, float]):
        """结束轮次"""
        epoch_time = time.time() - self.epoch_start_time
        
        # 记录指标
        self.training_metrics["train_losses"].append(train_loss)
        self.training_metrics["eval_losses"].append(eval_metrics.get("eval_loss", 0))
        self.training_metrics["eval_accuracies"].append(eval_metrics.get("eval_accuracy", 0))
        self.training_metrics["eval_f1_scores"].append(eval_metrics.get("eval_f1", 0))
        self.training_metrics["training_speed"].append(epoch_time)
        
        # 更新最佳模型
        current_f1 = eval_metrics.get("eval_f1", 0)
        if current_f1 > self.training_metrics["best_f1"]:
            self.training_metrics["best_f1"] = current_f1
            self.training_metrics["best_epoch"] = epoch
            
        # 记录日志
        self.logger.info(f"第 {epoch + 1} 轮完成 - 耗时: {epoch_time:.2f}s")
        self.logger.info(f"  训练损失: {train_loss:.4f}")
        self.logger.info(f"  验证损失: {eval_metrics.get('eval_loss', 0):.4f}")
        self.logger.info(f"  验证准确率: {eval_metrics.get('eval_accuracy', 0):.4f}")
        self.logger.info(f"  验证F1分数: {eval_metrics.get('eval_f1', 0):.4f}")
        
        # 保存指标
        self.save_metrics()
        
    def early_stop(self, epoch: int, reason: str):
        """早停"""
        self.training_metrics["early_stopped"] = True
        self.training_metrics["early_stopping_epoch"] = epoch
        self.logger.info(f"早停触发于第 {epoch + 1} 轮: {reason}")
        
    def end_training(self):
        """结束训练"""
        self.training_metrics["end_time"] = datetime.now().isoformat()
        
        # 计算总时间
        start_time = datetime.fromisoformat(self.training_metrics["start_time"])
        end_time = datetime.fromisoformat(self.training_metrics["end_time"])
        total_time = (end_time - start_time).total_seconds()
        
        # 训练摘要
        self.logger.info("="*50)
        self.logger.info("训练完成摘要:")
        self.logger.info(f"  总耗时: {total_time:.2f}s")
        self.logger.info(f"  完成轮数: {self.training_metrics['current_epoch'] + 1}/{self.training_metrics['total_epochs']}")
        self.logger.info(f"  最佳轮数: {self.training_metrics['best_epoch'] + 1}")
        self.logger.info(f"  最佳F1分数: {self.training_metrics['best_f1']:.4f}")
        self.logger.info(f"  最终训练损失: {self.training_metrics['train_losses'][-1]:.4f}")
        self.logger.info(f"  最终验证准确率: {self.training_metrics['eval_accuracies'][-1]:.4f}")
        
        if self.training_metrics["early_stopped"]:
            self.logger.info(f"  早停于第 {self.training_metrics['early_stopping_epoch'] + 1} 轮")
            
        self.logger.info("="*50)
        
        # 保存最终指标
        self.save_metrics()
        
    def save_metrics(self):
        """保存训练指标"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存指标失败: {e}")
            
    def load_metrics(self):
        """加载训练指标"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    self.training_metrics = json.load(f)
                    self.logger.info("成功加载历史训练指标")
                    return True
        except Exception as e:
            self.logger.error(f"加载指标失败: {e}")
        return False
        
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        if self.training_metrics["total_epochs"] == 0:
            return {"progress": 0, "status": "未开始"}
            
        progress = (self.training_metrics["current_epoch"] + 1) / self.training_metrics["total_epochs"]
        return {
            "progress": progress,
            "current_epoch": self.training_metrics["current_epoch"] + 1,
            "total_epochs": self.training_metrics["total_epochs"],
            "best_f1": self.training_metrics["best_f1"],
            "best_epoch": self.training_metrics["best_epoch"] + 1,
            "status": "早停" if self.training_metrics["early_stopped"] else "正常"
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.training_metrics["train_losses"]:
            return {"message": "暂无训练数据"}
            
        return {
            "轮次数": len(self.training_metrics["train_losses"]),
            "最佳F1分数": self.training_metrics["best_f1"],
            "最佳轮次": self.training_metrics["best_epoch"] + 1,
            "最终准确率": self.training_metrics["eval_accuracies"][-1] if self.training_metrics["eval_accuracies"] else 0,
            "平均训练速度": sum(self.training_metrics["training_speed"]) / len(self.training_metrics["training_speed"]) if self.training_metrics["training_speed"] else 0,
            "是否早停": self.training_metrics["early_stopped"]
        }

class EarlyStoppingMonitor:
    """早停监控器"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, score: float, epoch: int) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            self.logger.info(f"初始最佳分数: {score:.4f}")
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.logger.info(f"第 {epoch + 1} 轮更新最佳分数: {score:.4f}")
        else:
            self.counter += 1
            self.logger.info(f"第 {epoch + 1} 轮未改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"早停触发！最佳分数: {self.best_score:.4f}")
                
        return self.early_stop
        
    def _is_better(self, score: float, best_score: float) -> bool:
        """判断分数是否更好"""
        if self.mode == "max":
            return score > best_score + self.min_delta
        else:
            return score < best_score - self.min_delta
            
    def reset(self):
        """重置早停状态"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
