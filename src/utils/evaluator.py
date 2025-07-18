import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model_classifier):
        self.model = model_classifier
        
    def evaluate_on_dataset(self, df, sample_size=None):
        if sample_size:
            df = df.head(sample_size)
            
        predictions = []
        true_labels = []
        
        for _, row in df.iterrows():
            pred = self.model.predict(row['text'])
            predictions.append(pred['预测类别'])
            true_labels.append(row['label'])
        
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def test_samples(self, texts, expected_labels=None):
        results = []
        for i, text in enumerate(texts):
            pred = self.model.predict(text)
            result = {
                "text": text[:50] + "...",
                "prediction": pred
            }
            if expected_labels:
                result["expected"] = expected_labels[i]
                result["correct"] = pred['预测类别'] == expected_labels[i]
            results.append(result)
        return results
