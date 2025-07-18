import sys
sys.path.append('src')

from utils.config import MODEL_CONFIG
from utils.device_utils import DeviceManager
from model.bert_classifier import BertClassifier

def main():
    print("元白风格诗歌分类预测")
    
    # 显示设备信息
    print("设备信息:")
    device_manager = DeviceManager()
    device_manager.print_device_info()
    print()
    
    # 加载训练好的模型
    print("加载训练好的模型...")
    classifier = BertClassifier(MODEL_CONFIG["model_name"], MODEL_CONFIG["num_labels"])
    classifier.load_trained_model(MODEL_CONFIG["output_dir"])
    
    # 显示模型信息
    model_info = classifier.get_model_info()
    print(f"模型参数量: {model_info['总参数量']:,}")
    print(f"模型设备: {model_info['模型设备']}")
    print()
    
    # 测试样本
    test_texts = [
        "【臨都驛答夢得六言二首 一】揚子津頭月下，臨都驛裏燈前。昨日老於前日，去年春似今年。",
        "【歸雁二首 二】欲雪違胡地，先花別楚雲。却過清渭影，高起洞庭羣。塞北春陰暮，江南日色曛。傷弓流落羽，行斷不堪聞。",
        "【北亭招客】疎散郡丞同野客，幽閑官舍抵山家。春風北戶千莖竹，晚日東園一樹花。小醆吹醅嘗冷酒，深爐敲火炙新茶。"
    ]
    
    print("开始预测...")
    print("="*60)
    
    # 单个预测
    for i, text in enumerate(test_texts):
        result = classifier.predict(text)
        print(f"样本{i+1}: {text[:30]}...")
        print(f"预测结果: 元白风格概率={result['元白']:.3f}, 分类={'元白' if result['预测类别'] == 1 else '非元白'}")
        print(f"详细概率: 非元白={result['非元白']:.3f}, 元白={result['元白']:.3f}")
        print("-" * 60)
    
    # 批量预测示例
    print("\n批量预测示例:")
    batch_results = classifier.predict_batch(test_texts)
    for i, result in enumerate(batch_results):
        print(f"批量样本{i+1}: {'元白' if result['预测类别'] == 1 else '非元白'} (概率: {result['元白']:.3f})")
    
    print("\n预测完成！")

if __name__ == "__main__":
    main()
