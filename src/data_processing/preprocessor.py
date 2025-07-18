from transformers import BertTokenizer

class TextPreprocessor:
    def __init__(self, model_name, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
    
    def preprocess_datasets(self, train_ds, val_ds):
        train_ds = train_ds.map(self.tokenize_function, batched=True)
        val_ds = val_ds.map(self.tokenize_function, batched=True)
        
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        return train_ds, val_ds
