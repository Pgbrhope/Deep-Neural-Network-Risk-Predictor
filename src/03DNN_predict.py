import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# 載入模型
model = load_model('promising_patent_model.h5')

# 載入新資料
new_data_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\2021~2023.csv"
new_data = pd.read_csv(new_data_path)

# 填補缺失值
new_data = new_data.fillna(0)

# 處理 embedding 資料
embedding_feature = new_data["Embedding"].apply(lambda x: list(map(float, x[1:-1].split())))
embedding_df = pd.DataFrame(embedding_feature.tolist(), columns=[f"embedding_{i}" for i in range(100)])

# 合併 embedding 資料到特徵中
new_features = new_data[["Scope of rights","Scope of application","Size Of Contributors","Technology-base","Science Based","Applicant Type","Technological Scope","Commercial Scope","independent_claims","dependent_claims","COL","INV","Total Know-How"]].reset_index(drop=True)
new_features = pd.concat([new_features, embedding_df], axis=1)

# 標準化
scaler = StandardScaler()
new_features_scaled = scaler.fit_transform(new_features)

# 預測
predictions = model.predict(new_features_scaled)

# 將預測值填入 'Promising_score' 列
new_data['Promising_score'] = predictions

# 載入用於評估的資料集
evaluation_data_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\2000~2020.csv"
evaluation_data = pd.read_csv(evaluation_data_path)

# 将评估数据中的 'promising_patent' 列添加到新数据中
new_data['promising_patent'] = evaluation_data['promising_patent']

# 根据设定的threshold将预测结果转换为0或1
threshold = 0.756
new_data['promising_patent'] = (new_data['Promising_score'] > threshold).astype(int)

# 儲存預測結果到 CSV 檔案
output_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\predictions_2021_2023_adjusted.csv"
new_data.to_csv(output_path, index=False)

# 输出有希望的专利的真实数据
promising_patents = new_data[new_data['promising_patent'] == 1]
print("Promising Patents Data:")
print(promising_patents)
