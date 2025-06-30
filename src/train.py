import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import optuna

# 1. Загрузка данных
print("Loading data...")
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / 'data' / 'telco_churn.csv'

print(f"Data path: {data_path}")
data = pd.read_csv(data_path)

# 2. Предобработка данных
print("Processing TotalCharges column...")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
num_missing = data['TotalCharges'].isna().sum()
print(f"Found {num_missing} missing values in TotalCharges")
data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'] * data['tenure'])

print("Dropping customerID column...")
data = data.drop('customerID', axis=1)

# 3. Разделение данных
print("Splitting data...")
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Определение признаков
print("Defining features...")
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X_train.columns 
                        if col not in numerical_features and col != 'Churn']

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# 5. Препроцессинг данных
print("Preprocessing data...")
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    (StandardScaler(), numerical_features),
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Проверка данных
assert not np.isnan(X_train_processed).any(), "NaNs in training data"
assert not np.isinf(X_train_processed).any(), "Infs in training data"
print("Preprocessed data shape:", X_train_processed.shape)
print("Class distribution:", np.bincount(y_train))

# 6. Функция для оптимизации гиперпараметров
def objective(trial):
    # Проверка minority class для SMOTE
    minority_class_count = sum(y_train == 1)
    k_neighbors = min(5, minority_class_count - 1)
    
    if k_neighbors < 1:
        X_train_res, y_train_res = X_train_processed, y_train
        print(f"Trial {trial.number}: Using original data. Minority count: {minority_class_count}")
    else:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
        print(f"Trial {trial.number}: Applied SMOTE. New data shape: {X_train_res.shape}, minority class resampled to {sum(y_train_res==1)}")

    # Параметры для Optuna
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'max_leaves': trial.suggest_int('max_leaves', 0, 1000),
    }

    # Инициализация модели
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42,
        tree_method='hist',      # Используем CPU
        grow_policy='lossguide',  # Требуется для max_leaves
        **params
    )

    # Обучение с использованием early stopping
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test_processed, y_test)],
        verbose=0
    )

    # Прогноз для тестового набора
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

# 7. Оптимизация гиперпараметров
print("Optimizing hyperparameters with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 8. Результаты оптимизации
print("\nBest trial:")
trial = study.best_trial
print(f"  ROC-AUC: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 9. Обучение финальной модели на всех данных
print("\nTraining final model...")
minority_class_count = sum(y_train == 1)
k_neighbors = min(5, minority_class_count - 1)

if k_neighbors < 1:
    X_train_final, y_train_final = X_train_processed, y_train
    print("Using original data for final model")
else:
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_processed, y_train)
    print(f"Applied SMOTE for final model. New data shape: {X_train_final.shape}")

# Создаем финальную модель с лучшими параметрами
final_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    grow_policy='lossguide',
    **trial.params
)

# Обучаем на всех тренировочных данных
final_model.fit(X_train_final, y_train_final)

# 10. Оценка на тестовом наборе
y_pred_proba = final_model.predict_proba(X_test_processed)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nFinal model ROC-AUC on test set: {test_roc_auc:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Отчет о классификации и матрица ошибок
y_pred = final_model.predict(X_test_processed)
print("\n" + "="*50)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 2. ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(str(project_dir / 'reports/figures/roc_curve.png'))
plt.close()

# 3. Важность признаков
feature_importance = final_model.feature_importances_
feature_names = preprocessor.get_feature_names_out()

# Создаем DataFrame для визуализации
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()
plt.savefig(str(project_dir / 'reports/figures/feature_importance.png'))
plt.close()

print("\nTop 10 Features:")
print(importance_df.head(10))

# 11. Сохранение модели
model_path = project_dir / 'models' / 'xgboost_churn_model.json'
final_model.save_model(str(model_path))
print(f"Model saved to: {model_path}")

# Дополнительно: можно сохранить препроцессор
import joblib
preprocessor_path = project_dir / 'models' / 'preprocessor.pkl'
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to: {preprocessor_path}")