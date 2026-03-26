import heapq
import pandas as pd
import tensorflow as tf
import optuna
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.under_sampling import EditedNearestNeighbours
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def fill_nan_with_zero(df):
    return df.fillna(0)

def check_positive_samples(testY):
    num_positive_samples = np.sum(testY)  # Count positive samples
    return num_positive_samples

def objective(trial):
    random_state = int(time.time())  # 根據當前時間生成隨機數作為種子
    np.random.seed(trial.number)
    lr = trial.suggest_float("lr", 0.001, 0.01)
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 0.001, 0.01)
    decay_steps = trial.suggest_int("decay_steps", 10, 100)
    decay_rate = trial.suggest_float("decay_rate", 0.01, 1)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    batchSize = trial.suggest_int("batchSize", 10, 50)
    
    trainX, testX, trainY, testY = get_data(trial)  # 傳遞 random_state
    
    # Check positive samples in test set
    num_positive_samples = check_positive_samples(testY)
    
    model = create_model(trial)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule),
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])
    
    model.fit(trainX, trainY, epochs=100, batch_size=batchSize, shuffle=True, verbose=0)

    # Predict on the test set
    y_pred_proba = model.predict(testX)
    
    # Calculate F1 score
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(testY, y_pred_binary)
    
    # Calculate precision, recall, and accuracy
    precision = precision_score(testY, y_pred_binary)
    recall = recall_score(testY, y_pred_binary)
    accuracy = accuracy_score(testY, y_pred_binary)

    # Save F1 score, AUC-ROC score, and other metrics to trial user attributes
    trial.set_user_attr("f1_test", f1)
    trial.set_user_attr("best_precision", precision)
    trial.set_user_attr("best_recall", recall)
    trial.set_user_attr("max_f1_score", f1)
    trial.set_user_attr("best_accuracy", accuracy)
    
    # Save predicted labels and true labels to trial user attributes
    trial.set_user_attr("y_pred_binary", y_pred_binary.flatten())
    trial.set_user_attr("testY", pd.DataFrame(testY))

    return f1

def get_data(trial):
    neighbor = trial.suggest_int("neighbor", 1, 10)
    target = "promising_patent"
    file_path = r"C:\Users\User\Desktop\Barry\DNN_new\03Optuna\2000~2020.csv"
    dataset = pd.read_csv(file_path)

    embedding_feature = dataset["Embedding"].apply(lambda x: list(map(float, x[1:-1].split())))
    embedding_df = pd.DataFrame(embedding_feature.tolist(), columns=[f"embedding_{i}" for i in range(100)])

    features = dataset[["Scope of rights","Scope of application","Size Of Contributors","Technology-base","Science Based","Applicant Type","Technological Scope","Commercial Scope","independent_claims","dependent_claims","COL","INV","Total Know-How","promising_patent"]].reset_index(drop=True)
    features = pd.concat([features, embedding_df], axis=1)

    features = fill_nan_with_zero(features)

    labels = dataset[target].astype(np.float32)

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.2)

    smote = SMOTE(sampling_strategy='auto', k_neighbors=neighbor)
    trainX_resampled, trainY_resampled = smote.fit_resample(trainX, trainY)

    undersample = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=neighbor)
    trainX_resampled, trainY_resampled = undersample.fit_resample(trainX_resampled, trainY_resampled)

    return trainX_resampled, testX, trainY_resampled, testY

def create_model(trial):
    layer_num1 = trial.suggest_int("layer_num1", 5, 65)
    layer_num2 = trial.suggest_int("layer_num2", 5, 65)
    layer_num3 = trial.suggest_int("layer_num3", 5, 100)
    layer_num4 = trial.suggest_int("layer_num4", 5, 100)
    drop1 = trial.suggest_float("drop1", 0.01, 0.99)
    drop2 = trial.suggest_float("drop2", 0.01, 0.99)
    drop3 = trial.suggest_float("drop3", 0.01, 0.99)

    model = Sequential()
    model.add(Dense(layer_num1, input_dim=114, activation='relu'))
    model.add(Dropout(drop1))
    model.add(Dense(layer_num2, activation='relu'))  
    model.add(Dropout(drop2))
    model.add(Dense(layer_num3, activation='relu'))   
    model.add(Dropout(drop3))
    model.add(Dense(layer_num4, activation='relu'))         
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # 在 top_trials 中保存前三好的試驗，按照目標指標（F1 Score）進行排序
    top_trials = heapq.nlargest(3, study.get_trials(deepcopy=True), key=lambda trial: trial.value)

    # 創建一個 DataFrame 來保存結果
    results_df = pd.DataFrame(columns=["Trial", "F1 Score", "Precision", "Recall", "Accuracy", "Predictions", "True Labels"])
    # 印出前三個試驗的超參數和結果
    for i, trial in enumerate(top_trials):
        print(f"Trial {i + 1}:")
        print("  lr:", trial.params["lr"])
        print("  initial_learning_rate:", trial.params["initial_learning_rate"])
        print("  decay_steps:", trial.params["decay_steps"])
        print("  decay_rate:", trial.params["decay_rate"])
        print("  batchSize:", trial.params["batchSize"])
        print("  neighbor:", trial.params["neighbor"])
        print("  layer_num1:", trial.params["layer_num1"])
        print("  layer_num2:", trial.params["layer_num2"])
        print("  layer_num3:", trial.params["layer_num3"])
        print("  layer_num4:", trial.params["layer_num4"])
        print("  drop1:", trial.params["drop1"])
        print("  drop2:", trial.params["drop2"])
        print("  drop3:", trial.params["drop3"])
        print("  F1 Score:", trial.user_attrs["f1_test"])
        print("  Precision:", trial.user_attrs["best_precision"])
        print("  Recall:", trial.user_attrs["best_recall"])
        print("  Accuracy:", trial.user_attrs["best_accuracy"])
        print()

    # 遍歷 top_trials，保存結果到 DataFrame
    for i, trial in enumerate(top_trials):
        trial_number = i + 1
        f1 = trial.user_attrs["f1_test"]
        precision = trial.user_attrs["best_precision"]
        recall = trial.user_attrs["best_recall"]
        accuracy = trial.user_attrs["best_accuracy"]
        predictions = trial.user_attrs["y_pred_binary"]
        true_labels = trial.user_attrs["testY"]
        
        # 展平Predictions列表
        predictions_flat = predictions.flatten()

        # 將True Labels轉換為DataFrame
        true_labels_df = pd.DataFrame(true_labels)

        # 將展平後的Predictions和True Labels合併成一個DataFrame
        trial_results_df = pd.DataFrame({
            "Trial": [trial_number] * len(predictions_flat),
            "F1 Score": [f1] * len(predictions_flat),
            "Precision": [precision] * len(predictions_flat),
            "Recall": [recall] * len(predictions_flat),
            "Accuracy": [accuracy] * len(predictions_flat),
            "Predictions": predictions_flat,
            "True Labels": true_labels_df.values.flatten()
        })

        results_df = pd.concat([results_df, trial_results_df], ignore_index=True)

    # 將結果保存到 CSV 文件
    results_df.to_csv("top_trials_predictions.csv", index=False)
