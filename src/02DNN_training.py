import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import matplotlib.pyplot as plt

# 載入資料
file_path = r"C:\Users\user\Desktop\IAN\IAN_接續\barry_patent取代\2000~2020_replace_patent.csv"
dataset = pd.read_csv(file_path)

# 定義填補缺失值的函數
def fill_nan_with_zero(df):
    return df.fillna(0)

# 處理 embedding 資料
embedding_feature = dataset["Embedding"].apply(lambda x: list(map(float, x[1:-1].split())))
embedding_df = pd.DataFrame(embedding_feature.tolist(), columns=[f"embedding_{i}" for i in range(100)])

# 合併 embedding 資料到特徵中
features = dataset[["Scope of rights","Scope of application","Size Of Contributors","Technology-base","Science Based","Applicant Type","Technological Scope","Commercial Scope","independent_claims","dependent_claims","COL","INV","Total Know-How"]].reset_index(drop=True)
features = pd.concat([features, embedding_df], axis=1)

features = fill_nan_with_zero(features)

labels = dataset["promising_patent"].astype(np.float32)

# 標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用 SMOTEENN 進行過取樣和欠取樣
smote = SMOTE(sampling_strategy='auto', k_neighbors=2)
features_resampled, labels_resampled = smote.fit_resample(features_scaled, labels)

# 過取樣後的欠取樣策略
undersample = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=2)
features_resampled, labels_resampled = undersample.fit_resample(features_resampled, labels_resampled)

# 將資料分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42)

# 定義超參數
lr = 0.0034771808021403557
initial_learning_rate = 0.00941964634700437
decay_steps = 17
decay_rate = 0.9190058076381927
batchSize = 36
neighbor = 2
layer_num1 = 16
layer_num2 = 38
layer_num3 = 57
layer_num4 = 20
drop1 = 0.24386507033568422
drop2 = 0.3187067982638068
drop3 = 0.17326916770633685

# 設定學習率衰減
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# 建立模型
model = Sequential()
model.add(Dense(layer_num1, input_dim=113, activation='relu'))
model.add(Dropout(drop1))
model.add(Dense(layer_num2, activation='relu'))
model.add(Dropout(drop2))
model.add(Dense(layer_num3, activation='relu'))
model.add(Dropout(drop3))
model.add(Dense(layer_num4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule),
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=100, batch_size=batchSize, shuffle=True, verbose=0)

# 預測測試集
predictions_test = model.predict(X_test)

# 將預測轉換為二元標籤
binary_predictions_test = (predictions_test > 0.5).astype(int)

# 計算測試集的 precision、recall 和 F1 score
precision_test = precision_score(y_test, binary_predictions_test)
recall_test = recall_score(y_test, binary_predictions_test)
f1_test = f1_score(y_test, binary_predictions_test)

# 印出結果
print("Test Precision:", precision_test)
print("Test Recall:", recall_test)
print("Test F1 Score:", f1_test)

# 新資料預測
# 載入新資料
new_data_path = r"C:\Users\user\Desktop\IAN\IAN_接續\Barry2024_DNN_new_複製\03Optuna(直接開這個)\2021~2023.csv"
new_data = pd.read_csv(new_data_path)

# 做與訓練時相同的前處理
# 填補缺失值
new_data = new_data.fillna(0)

# 處理 embedding 資料
embedding_feature = new_data["Embedding"].apply(lambda x: list(map(float, x[1:-1].split())))
embedding_df = pd.DataFrame(embedding_feature.tolist(), columns=[f"embedding_{i}" for i in range(100)])

# 合併 embedding 資料到特徵中
new_features = new_data[["Scope of rights","Scope of application","Size Of Contributors","Technology-base","Science Based","Applicant Type","Technological Scope","Commercial Scope","independent_claims","dependent_claims","COL","INV","Total Know-How"]].reset_index(drop=True)
new_features = pd.concat([new_features, embedding_df], axis=1)

# 添加 'promising_patent' 列並初始化為空值
new_features['promising_patent'] = np.nan

# 重新排列新資料集的特徵，使其與訓練模型時使用的特徵順序相同
new_features = new_features[features.columns]

# 標準化
new_features_scaled = scaler.transform(new_features)

# 預測
predictions = model.predict(new_features_scaled)

# 使用閥值進行二元化預測
threshold =0.03 # 從閥值分析圖上得到的閥值
binary_predictions = (predictions > threshold).astype(int)

# 將預測值填入 'promising_patent' 列
new_data['promising_patent'] = binary_predictions

# 儲存預測結果到 CSV 檔案
output_path = r"C:\Users\user\Desktop\IAN\IAN_接續\predictions_2021_2023.csv"
new_data.to_csv(output_path, index=False)

# 儲存模型
model.save('promising_patent_model.h5')


# 閥值分析
thresholds = np.arange(0, 1, 0.05)
promising_patents_count = []

for threshold in thresholds:
    binary_predictions_threshold = (predictions > threshold).astype(int)
    promising_patents_count.append(np.sum(binary_predictions_threshold))

# 繪製閥值分析圖
plt.plot(thresholds, promising_patents_count)
plt.xlabel('Threshold')
plt.ylabel('Promising Patents Count')
plt.title('Threshold Analysis')
plt.grid(True)
plt.show()
