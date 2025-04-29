import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# 1. 加载数据
data = pd.read_csv('cgls_0413_V2.csv')  # 替换为您的文件路径

# 2. 数据预处理
# 选择特征和标签
features = data[['m', 'n'] + [f'Norm{i}/Norm0' for i in range(1, 6)]]
labels = data['flag']

# 将科学计数法转换为浮点数（如果pandas没有自动转换）
for col in features.columns:
    if features[col].dtype == object:
        features[col] = features[col].astype(float)

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 编码标签 (1,2,3 -> 0,1,2)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# 将标签转换为one-hot编码
one_hot_labels = to_categorical(encoded_labels)

# 3. 准备LSTM输入数据
# 将数据重塑为LSTM需要的3D格式 [samples, timesteps, features]
X = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
y = one_hot_labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# 6. 评估模型
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_.astype(str)))

# 7. 保存模型
model.save('lstm_classifier.h5')

# 绘制模型结构图并保存
plot_model(model, to_file='lstm_model_structure.png', show_shapes=True, show_layer_names=True)

# 8. 添加测试样例功能
def test_single_sample(model, scaler, label_encoder, sample_data):
    """
    测试单个样本的预测结果
    
    参数:
        model: 训练好的模型
        scaler: 用于特征标准化的scaler
        label_encoder: 用于标签编码的encoder
        sample_data: 字典形式的数据样本，包含所有需要的特征
    """
    # 准备特征数据
    features = ['m', 'n'] + [f'Norm{i}/Norm0' for i in range(1, 6)]
    sample_features = np.array([sample_data[feat] for feat in features]).reshape(1, -1)
    
    # 标准化特征
    scaled_sample = scaler.transform(sample_features)
    
    # 重塑为LSTM输入格式
    lstm_input = scaled_sample.reshape(1, 1, -1)
    
    # 进行预测
    prediction = model.predict(lstm_input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]
    
    # 输出预测结果
    print("\n测试样例预测结果:")
    print(f"输入特征: {sample_data}")
    print(f"预测概率分布: {prediction[0]}")
    print(f"预测类别: {predicted_label} (原始编码: {predicted_class[0]})")
    
    return predicted_label

# 9. 创建一个测试样例并预测
# 从原始数据中随机选取一个样本作为测试样例
sample_index = np.random.randint(0, len(data))
sample_data = {
    'm': data.loc[sample_index, 'm'],
    'n': data.loc[sample_index, 'n'],
    **{f'Norm{i}/Norm0': data.loc[sample_index, f'Norm{i}/Norm0'] for i in range(1, 6)}
}

# 获取真实标签
true_label = data.loc[sample_index, 'flag']

# 进行预测
predicted_label = test_single_sample(model, scaler, label_encoder, sample_data)

# 输出真实标签
print(f"真实类别: {true_label}")
print(f"预测是否正确: {true_label == predicted_label}")