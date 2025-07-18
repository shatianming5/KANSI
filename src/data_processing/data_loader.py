import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, train_file, val_file, sample_size=None):
        df_train = pd.read_csv(train_file)[[self.config["text_column"], self.config["label_column"]]]
        df_val = pd.read_csv(val_file)[[self.config["text_column"], self.config["label_column"]]]
        
        if sample_size:
            df_train = df_train.head(sample_size)
            df_val = df_val.head(sample_size // 5)
            
        return df_train, df_val
    
    def create_datasets(self, df_train, df_val):
        train_ds = Dataset.from_pandas(df_train)
        val_ds = Dataset.from_pandas(df_val)
        return train_ds, val_ds
    
    def get_data_stats(self, df):
        stats = {
            "总样本数": len(df),
            "标签分布": df[self.config["label_column"]].value_counts().to_dict(),
            "文本长度统计": {
                "平均长度": df[self.config["text_column"]].str.len().mean(),
                "最大长度": df[self.config["text_column"]].str.len().max(),
                "最小长度": df[self.config["text_column"]].str.len().min()
            }
        }
        return stats
