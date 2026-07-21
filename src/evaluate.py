import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import shap
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    print(f"\n{'='*50}")
    print(f"EVALUACIÓN: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != 'Modelo':
            print(f"{k}: {v:.4f}")

    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name='Model', save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz guardada en: {save_path}")
    plt.show()


def plot_roc_curve(y_test, y_proba_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    for model_name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate (1 - Especificidad)')
    plt.ylabel('True Positive Rate (Sensibilidad)')
    plt.title('Curva ROC - Comparación de Modelos')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curva ROC guardada en: {save_path}")
    plt.show()


def plot_precision_recall_curve(y_test, y_proba_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    for model_name, y_proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=model_name, linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall - Comparación de Modelos')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curva PR guardada en: {save_path}")
    plt.show()


def compare_models(metrics_list, save_path=None):
    df = pd.DataFrame(metrics_list)
    df = df.round(4)

    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    df_melted = df.melt(id_vars=['Modelo'], var_name='Métrica', value_name='Valor')
    sns.barplot(data=df_melted, x='Métrica', y='Valor', hue='Modelo', ax=ax)
    plt.title('Comparación de Métricas entre Modelos')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparación guardada en: {save_path}")
    plt.show()

    return df


def plot_feature_importance(model, feature_names, model_name='Model', top_n=15, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"No se puede obtener importancia de variables para {model_name}")
        return

    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp, y='Feature', x='Importance', palette='viridis')
    plt.title(f'Top {top_n} Variables más Importantes - {model_name}')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Importancia guardada en: {save_path}")
    plt.show()

    return feat_imp


def explain_with_shap(model, X_test, feature_names, model_type='tree', save_path=None):
    print("\n=== ANÁLISIS SHAP ===")

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_sample = X_test_df.sample(min(100, len(X_test_df)), random_state=42)

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.Explainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_summary_bar.png'), dpi=150, bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return shap_values, X_sample
