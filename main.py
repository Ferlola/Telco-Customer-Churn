import sys
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

from src.preprocessing import preprocess_data, save_preprocessing_artifacts
from src.feature_engineering import engineer_features
from src.train_models import train_all_models
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, compare_models, plot_feature_importance
)
from src.retention import predict_batch, retention_summary

print("=" * 60)
print("TELCO CUSTOMER CHURN - PIPELINE COMPLETO")
print("=" * 60)

print("\n[1/5] Preprocesamiento de datos...")
prep_result = preprocess_data(
    filepath='data/raw/Telco_customer_churn.xlsx',
    test_size=0.2,
    random_state=42,
    apply_smote_flag=True
)

df_clean = prep_result['df_clean']
df_feat = engineer_features(df_clean)

X_train = prep_result['X_train']
X_test = prep_result['X_test']
y_train = prep_result['y_train']
y_test = prep_result['y_test']
feature_names = prep_result['feature_names']

save_preprocessing_artifacts(prep_result)

import joblib
joblib.dump(prep_result, 'data/processed/preprocessed_data.pkl')
print("Datos preprocesados guardados en data/processed/preprocessed_data.pkl")

print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

print("\n[2/5] Entrenamiento de modelos...")
models = train_all_models(X_train, y_train, save=True)

print("\n[3/5] Evaluación de modelos...")
metrics_list = []
y_proba_dict = {}
 
for name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']:
    model = models[name]
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, name)
    metrics_list.append(metrics)
    y_proba_dict[name] = y_proba

    plot_confusion_matrix(
        y_test, y_pred, name,
        save_path=f'reports/figures/cm_{name.lower().replace(" ", "_")}.png'
    )

print("\n[4/5] Comparación de modelos...")
comparison_df = compare_models(
    metrics_list,
    save_path='reports/figures/model_comparison.png'
)
comparison_df.to_csv('reports/metrics/model_comparison.csv', index=False)

plot_roc_curve(y_test, y_proba_dict, save_path='reports/figures/roc_curves.png')
plot_precision_recall_curve(y_test, y_proba_dict, save_path='reports/figures/pr_curves.png')

best_model_name = comparison_df.sort_values('ROC-AUC', ascending=False).iloc[0]['Modelo']
best_model = models[best_model_name]
print(f"\nMejor modelo: {best_model_name}")

feat_imp_df = plot_feature_importance(
    best_model, feature_names, best_model_name, top_n=15,
    save_path='reports/figures/best_model_feature_importance.png'
)
if feat_imp_df is not None:
    feat_imp_df.to_csv('reports/metrics/feature_importance.csv', index=False)

print("\n[5/5] Sistema de retención...")
predictions_df = predict_batch(best_model, X_test)
predictions_df.to_csv('reports/metrics/predictions.csv', index=False)

retention_summary(predictions_df)

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 60)
print("\nDashboard: streamlit run app/app.py")
