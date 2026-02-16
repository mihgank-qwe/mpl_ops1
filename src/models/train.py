"""Обучение модели с логированием в MLflow"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV

from src.models.pipeline import create_pipeline

# Колонки признаков из обработанных данных
NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "PAY_MEAN",
    "PAY_MAX",
    "PAY_MIN",
    "BILL_AMT_MEAN",
    "BILL_AMT_MAX",
    "PAY_AMT_MEAN",
    "PAY_AMT_SUM",
    "PAY_TO_BILL_RATIO",
]
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]
TARGET_COL = "default.payment.next.month"


def load_data():
    """Метод загружает train и test"""
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def plot_and_save_roc_curve(y_test, y_pred_proba, output_path):
    """Метод строит ROC-кривую и сохраняет в файл"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Доля ложных срабатываний")
    plt.ylabel("Доля истинных срабатываний")
    plt.title("ROC-кривая")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri("file:///./mlruns")
    mlflow.set_experiment("Credit_Default_Prediction")

    with mlflow.start_run():
        pipeline = create_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

        # Подбор гиперпараметров
        param_dist = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.01, 0.1],
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=3,
            cv=2,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_

        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_params(search.best_params_)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        # Сохранение ROC-кривой
        project_root = Path(__file__).resolve().parents[2]
        roc_path = project_root / "roc_curve.png"
        plot_and_save_roc_curve(y_test, y_pred_proba, roc_path)
        mlflow.log_artifact(str(roc_path))

        # Сохранение метрик
        metrics = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        metrics_path = project_root / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Сохранение модели
        model_path = project_root / "models" / "credit_default_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

        mlflow.sklearn.log_model(
            pipeline, "model", registered_model_name="CreditDefaultModel"
        )

        print(
            f"Metrics: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )
        print(f"Модель сохранена: {model_path}")


if __name__ == "__main__":
    main()
