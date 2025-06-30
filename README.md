# Telco Customer Churn Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Прогнозирование оттока клиентов телеком-компании с использованием XGBoost и оптимизации гиперпараметров через Optuna

# Ключевые особенности

- Предобработка данных: обработка пропусков, кодирование категориальных признаков
- Оптимизация гиперпараметров с помощью Optuna (100+ trials)
- Балансировка классов через SMOTE
- Интерпретация модели: анализ важности признаков
- Сохранение модели и препроцессора

Проект решает задачу бинарной классификации для предсказания оттока клиентов с использованием XGBoost. Лучшая модель достигла F1-score 0.6243 на тестовых данных

## Инструкция по запуску
```bash
git clone https://github.com/DmitryKim1/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py


## Подготовка данных
1. Скачайте dataset с [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Поместите файл `WA_Fn-UseC_-Telco-Customer-Churn.csv` в папку `data/`

## Пример использования
После обучения модели вы можете использовать её для предсказаний:
```python
from src.predict import predict_churn

data = {...}  # данные клиента
prediction = predict_churn(data)
print(f"Вероятность оттока: {prediction:.2%}")
