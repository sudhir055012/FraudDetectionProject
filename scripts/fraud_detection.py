import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import os
import joblib

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directories for dissertation evidence
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)


# --- UNIVERSAL WRAPPER FOR CATBOOST COMPATIBILITY ---
class FixedCatBoostClassifier(CatBoostClassifier, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_tags(self):
        return {"allow_nan": True, "pairwise": False}

    def __sklearn_tags__(self):
        class MockTags:
            def __init__(self):
                self.estimator_type = "classifier"
                self.input_tags = type('InputTags', (object,), {
                    'allow_nan': True, 'sparse': False, 'pairwise': False,
                    'string': False, 'dict': False
                })()
                self.target_tags = type('TargetTags', (object,), {
                    'single_output': True, 'multi_output': False
                })()

        return MockTags()


# 1️⃣ Visualization: Class Distribution
def plot_class_distribution(y, stage="Original"):
    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.title(f"Class Distribution ({stage})")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Count")
    plt.savefig(f"../results/class_distribution_{stage.lower()}.png")
    plt.close()


# 2️⃣ Visualization: Transaction Amount Distribution
def plot_amount_distribution(df):
    plt.figure(figsize=(6, 4))
    # Using log_scale because transaction amounts vary wildly
    sns.histplot(df['Amount'], bins=50, log_scale=True, kde=True)
    plt.title("Transaction Amount Distribution (Log Scale)")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.savefig("../results/amount_distribution.png")
    plt.close()


def load_and_preprocess(filepath):
    print("--- 1. Data Collection & Preprocessing ---")
    df = pd.read_csv(filepath)

    # Generate Amount Distribution Plot
    plot_amount_distribution(df)

    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    X = df.drop(['Class', 'Time', 'Amount'], axis=1)
    y = df['Class']

    # Initial Class Distribution Plot
    plot_class_distribution(y, stage="Initial")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 2. Applying SMOTE (Balancing Majority/Minority Classes) ---")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Distribution Plot after SMOTE to show the balance achieved
    plot_class_distribution(y_train_res, stage="Post-SMOTE")

    return X_train_res, X_test, y_train_res, y_test


def run_experiment(name, model, X_train, y_train, X_test, y_test):
    print(f"\n>>> Training Model: {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Save full reports
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f'../results/{name}_full_report.csv')

    auprc = average_precision_score(y_test, y_probs)

    # 3️⃣ Visualization: Precision-Recall Curve (REQUIRED for Fraud)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve: {name}")
    plt.legend()
    plt.savefig(f"../results/{name}_pr_curve.png")
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(f'../results/{name}_cm.png')
    plt.close()

    joblib.dump(model, f'../models/{name}_model.pkl')
    return {"Model": name, "AUPRC": auprc}


def build_stacking_ensemble():
    level0 = [
        ('xgb',
         XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1)),
        ('cat', FixedCatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0))
    ]
    return StackingClassifier(estimators=level0, final_estimator=LogisticRegression(), cv=3)


if __name__ == "__main__":
    DATA_PATH = '../data/creditcard.csv'

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
        performance_results = []

        # Baselines and Advanced Ensemble
        performance_results.append(
            run_experiment("Baseline_Logistic", LogisticRegression(max_iter=1000), X_train, y_train, X_test, y_test))
        performance_results.append(
            run_experiment("Baseline_RandomForest", RandomForestClassifier(n_estimators=100, random_state=42), X_train,
                           y_train, X_test, y_test))

        ensemble_model = build_stacking_ensemble()
        performance_results.append(
            run_experiment("Advanced_Stacking", ensemble_model, X_train, y_train, X_test, y_test))

        # 4️⃣ Visualization: Final Model Comparison Bar Chart
        summary_df = pd.DataFrame(performance_results)
        summary_df.to_csv('../results/model_comparison_summary.csv', index=False)

        plt.figure(figsize=(7, 5))
        sns.barplot(x="Model", y="AUPRC", data=summary_df, palette="viridis")
        plt.title("Comparative Performance: Model vs. AUPRC")
        plt.ylim(0, 1)
        plt.savefig("../results/model_comparison_barplot.png")
        plt.close()

        # 5️⃣ XAI: SHAP Explanations
        print("\n--- Generating SHAP Explanations ---")
        xgb_internal = ensemble_model.named_estimators_['xgb']
        explainer = shap.TreeExplainer(xgb_internal)
        shap_values = explainer.shap_values(X_test[:50])

        plt.figure()
        shap.summary_plot(shap_values, X_test[:50], show=False)
        plt.savefig('../results/shap_summary_plot.png')
        plt.close()  # Added to prevent memory leaks

        print("\nSuccess! Evidence Portfolio generated in '../results/' and '../models/'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")