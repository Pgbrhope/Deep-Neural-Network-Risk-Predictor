import pandas as pd

# 讀取 CSV 檔案
file_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\predictions_2021_2023_adjusted.csv"

data = pd.read_csv(file_path)

# 計算資料總數
total_records = data.shape[0]

# 計算 promising_patent 為 1 的數量
promising_patent_count = data['promising_patent'].sum()

# 顯示結果
print("資料總數:", total_records)
print("promising_patent 為 1 的數量:", promising_patent_count)

# 篩選出 promising_patent 為 1 的資料
promising_patent_data = data[data['promising_patent'] == 1]

# 儲存篩選後的資料到 CSV 檔案
output_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\promising_patent_2021~2023.csv"
promising_patent_data.to_csv(output_path, index=False)

print("已生成 promising_patent_2021~2023.csv 檔案")
