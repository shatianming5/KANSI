import torch
import logging

class DeviceManager:
    def __init__(self):
        self.device = self._get_best_device()
        self.device_info = self._get_device_info()
        
    def _get_best_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_device_info(self):
        info = {'device_type': self.device.type}
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            })
        elif self.device.type == 'mps':
            info.update({
                'gpu_name': 'Apple Silicon GPU',
                'gpu_count': 1
            })
        else:
            info.update({
                'cpu_name': 'CPU',
                'cpu_count': torch.get_num_threads()
            })
        
        return info
    
    def get_device(self):
        return self.device
    
    def get_device_info(self):
        return self.device_info
    
    def print_device_info(self):
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU名称: {self.device_info.get('gpu_name', 'Unknown')}")
            print(f"GPU数量: {self.device_info.get('gpu_count', 0)}")
            print(f"总内存: {self.device_info.get('memory_total', 0):.2f} GB")
            print(f"可用内存: {self.device_info.get('memory_available', 0):.2f} GB")
        elif self.device.type == 'mps':
            print(f"GPU名称: {self.device_info.get('gpu_name', 'Unknown')}")
        else:
            print(f"CPU线程数: {self.device_info.get('cpu_count', 1)}")
    
    def move_to_device(self, tensor_or_model):
        return tensor_or_model.to(self.device)
    
    def clear_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
