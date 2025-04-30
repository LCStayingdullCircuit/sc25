import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

def load_preprocessing_components(data_path):
    """
    加载原始数据并创建预处理组件（标准化器和标签编码器）
    
    参数:
        data_path: 原始训练数据的CSV文件路径
    
    返回:
        scaler: 训练好的StandardScaler
        label_encoder: 训练好的LabelEncoder
    """
    # 加载原始数据
    data = pd.read_csv(data_path)
    
    # 获取特征和标签
    features = data[['m', 'n'] + [f'Norm{i}/Norm0' for i in range(1, 6)]]
    labels = data['flag']
    
    # 为特征创建并训练标准化器
    scaler = StandardScaler()
    scaler.fit(features)
    
    # 为标签创建并训练编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    return scaler, label_encoder

def predict_single_sample(model_path, scaler, label_encoder, m, n, norm1_norm0, norm2_norm0, norm3_norm0, norm4_norm0, norm5_norm0):
    """
    使用保存的模型预测单个样本
    
    参数:
        model_path: 保存的模型文件路径
        scaler: 训练好的StandardScaler
        label_encoder: 训练好的LabelEncoder
        m, n: 特征值
        norm1_norm0 到 norm5_norm0: 归一化特征值
    
    返回:
        predicted_label: 预测的标签
        probabilities: 每个类别的预测概率
    """
    # 加载模型
    model = load_model(model_path)
    
    # 准备输入特征
    sample_features = np.array([m, n, norm1_norm0, norm2_norm0, norm3_norm0, norm4_norm0, norm5_norm0]).reshape(1, -1)
    
    # 标准化特征
    scaled_sample = scaler.transform(sample_features)
    
    # 重塑为LSTM输入格式 [samples, timesteps, features]
    lstm_input = scaled_sample.reshape(1, 1, -1)
    
    # 进行预测
    prediction = model.predict(lstm_input, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]
    
    return predicted_label, prediction[0]

# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用保存的LSTM模型进行预测')
    parser.add_argument('--model_path', type=str, default='lstm_classifier.h5', help='保存的模型路径')
    parser.add_argument('--data_path', type=str, default='cgls_0413_V2.csv', help='原始训练数据路径（用于创建预处理器）')
    parser.add_argument('values', type=float, nargs=7, 
                        help='按顺序输入7个特征值: m n norm1 norm2 norm3 norm4 norm5')
    
    args = parser.parse_args()
    
    # 获取7个特征值
    m, n, norm1, norm2, norm3, norm4, norm5 = args.values
    
    # 加载预处理组件
    scaler, label_encoder = load_preprocessing_components(args.data_path)
    
    # 进行预测
    predicted_label, probabilities = predict_single_sample(
        args.model_path, scaler, label_encoder,
        m, n, norm1, norm2, norm3, norm4, norm5
    )
    
    # 输出结果
    print("\n预测结果:")
    print(f"预测类别: {predicted_label}")
    print(f"预测概率分布:")
    
    for i, prob in enumerate(probabilities):
        original_label = label_encoder.inverse_transform([i])[0]
        print(f"  类别 {original_label}: {prob:.4f}")