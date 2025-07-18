import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from utils.device_utils import DeviceManager

class BertClassifier:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.device_manager = DeviceManager()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """加载预训练模型"""
        self.logger.info(f"正在加载模型: {self.model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
        # 将模型移动到GPU
        self.model = self.device_manager.move_to_device(self.model)
        
        self.logger.info(f"模型已加载到设备: {self.device_manager.device}")
        
        return self.model, self.tokenizer
    
    def load_trained_model(self, model_path):
        """加载已训练的模型"""
        self.logger.info(f"正在加载训练好的模型: {model_path}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        
        # 将模型移动到GPU
        self.model = self.device_manager.move_to_device(self.model)
        
        # 设置为评估模式
        self.model.eval()
        
        self.logger.info(f"模型已加载到设备: {self.device_manager.device}")
        
        return self.model, self.tokenizer
    
    def predict(self, text, max_length=128):
        """预测单个文本"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载")
            
        # 确保模型在正确设备上
        if self.model.device != self.device_manager.device:
            self.model = self.device_manager.move_to_device(self.model)
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        )
        
        # 将输入移动到同一设备
        inputs = {k: self.device_manager.move_to_device(v) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        return {
            "非元白": float(probs[0]),
            "元白": float(probs[1]),
            "预测类别": int(1 if probs[1] > 0.5 else 0)
        }
    
    def predict_batch(self, texts, max_length=128, batch_size=16):
        """批量预测文本"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载")
            
        # 确保模型在正确设备上
        if self.model.device != self.device_manager.device:
            self.model = self.device_manager.move_to_device(self.model)
            
        self.model.eval()
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            # 将输入移动到同一设备
            inputs = {k: self.device_manager.move_to_device(v) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
            batch_results = []
            for j, prob in enumerate(probs):
                batch_results.append({
                    "text": batch_texts[j],
                    "非元白": float(prob[0]),
                    "元白": float(prob[1]),
                    "预测类别": int(1 if prob[1] > 0.5 else 0)
                })
            
            results.extend(batch_results)
            
            # 清理缓存
            if i % (batch_size * 10) == 0:
                self.device_manager.clear_cache()
                
        return results
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return "模型未加载"
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "模型名称": self.model_name,
            "总参数量": total_params,
            "可训练参数量": trainable_params,
            "模型设备": str(self.model.device),
            "当前设备": str(self.device_manager.device)
        }
