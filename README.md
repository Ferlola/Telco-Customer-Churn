# Telco Customer Churn Prediction System

## Sistema Inteligente de Predicción de Abandono y Plan de Retención

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Descripción del Proyecto

Proyecto completo de Ciencia de Datos siguiendo la metodología **CRISP-DM** para predecir el abandono de clientes (*churn*) en una empresa de telecomunicaciones. El sistema permite identificar clientes en riesgo, clasificar su nivel de amenaza y recomendar automáticamente estrategias de retención personalizadas.

**Dataset:** IBM Telco Customer Churn (7,043 clientes, 33 variables)

---

## 🏗️ Arquitectura del Proyecto

```
Telco_Customer_Churn/
│
├── data/
│   ├── raw/                      # Datos originales
│   └── processed/                # Datos preprocesados
│
├── notebooks/                    # Jupyter Notebooks (CRISP-DM)
│   ├── 01_EDA.ipynb              # Análisis Exploratorio
│   ├── 02_Preprocessing.ipynb    # Preprocesamiento y FE
│   ├── 03_Modeling.ipynb         # Entrenamiento de modelos
│   └── 04_Evaluation.ipynb       # Evaluación y conclusiones
│
├── src/                          # Código fuente modular
│   ├── preprocessing.py          # Limpieza, encoding, scaling, SMOTE
│   ├── feature_engineering.py    # Creación de nuevas variables
│   ├── train_models.py           # Entrenamiento con GridSearchCV
│   ├── evaluate.py               # Métricas, SHAP, visualizaciones
│   └── retention.py              # Sistema de retención basado en riesgo
│
├── models/                       # Modelos entrenados (.pkl)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── reports/
│   ├── figures/                  # Gráficos generados
│   └── metrics/                  # CSV con métricas
│
├── app/
│   └── app.py                    # Dashboard interactivo Streamlit
│
├── requirements.txt
└── README.md
```

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Construir un modelo de clasificación supervisada que prediga la probabilidad de abandono de clientes y diseñar un plan de retención basado en los resultados.

### Objetivos Específicos
1. Identificar los factores más influyentes en el abandono de clientes
2. Comparar múltiples algoritmos de clasificación (Regresión Logística, Árbol de Decisión, Random Forest, XGBoost)
3. Seleccionar el mejor modelo según métricas de rendimiento e interpretabilidad
4. Generar recomendaciones de retención personalizadas según el nivel de riesgo

---

## 📊 Metodología CRISP-DM

### Fase 1: Comprensión del Negocio
- **Problema:** Alta tasa de abandono de clientes (~27%) en empresa de telecomunicaciones
- **Impacto económico:** Cada cliente perdido representa una pérdida de ingresos recurrentes y costo de adquisición de nuevos clientes
- **Beneficio:** Identificar clientes en riesgo permite tomar acciones proactivas de retención
- **Objetivo:** Reducir el churn en al menos un 15% mediante intervenciones tempranas

### Fase 2: Comprensión de los Datos
- 7,043 registros, 33 variables
- Variable objetivo: Churn Value (0=No, 1=Sí) con 73.5% / 26.5% (desbalanceado)
- Variables predictoras: datos demográficos, servicios contratados, cargos, antigüedad, etc.
- Sin valores nulos en variables predictoras

### Fase 3: Preprocesamiento
- Conversión de Total Charges a numérico
- Eliminación de columnas no predictivas (CustomerID, coordenadas, etc.)
- One-Hot Encoding para variables categóricas multiclase
- Label Encoding para variables binarias
- StandardScaler para variables numéricas
- SMOTE para balanceo de clases

### Fase 4: Ingeniería de Características
- **Avg Monthly Spend:** Gasto promedio mensual (Total Charges / Tenure)
- **Tenure Group:** Categorización de la antigüedad (0-6m, 6-12m, 1-2a, 2-4a, 4-6a, 6+a)
- **Num Services:** Cantidad de servicios contratados
- **Premium Customer:** Cliente con alto gasto y larga antigüedad
- **New Customer:** Cliente nuevo (tenure <= 6 meses)
- **Engagement Score:** Proporción de servicios contratados

### Fase 5: Modelado
4 algoritmos entrenados con optimización de hiperparámetros mediante GridSearchCV (5-fold CV).

### Fase 6: Evaluación
Comparación de modelos usando Accuracy, Precision, Recall, F1-Score y ROC-AUC.

### Fase 7: Interpretabilidad
- **SHAP:** Análisis de contribución de variables
- **Feature Importance:** Importancia de variables para cada modelo
- **Dependence Plots:** Relación entre variables y predicción

---

## 🚀 Modelos Entrenados

| Modelo | Técnica de Optimización |
|--------|------------------------|
| Regresión Logística | class_weight='balanced' |
| Árbol de Decisión | GridSearchCV: max_depth, min_samples_split, criterion |
| Random Forest | GridSearchCV: n_estimators, max_depth, max_features |
| XGBoost | GridSearchCV: learning_rate, max_depth, n_estimators, subsample |

---

## 🏆 Resultados Esperados

El mejor modelo (XGBoost) alcanza las siguientes métricas:

| Métrica | Valor |
|---------|-------|
| Accuracy | ~80% |
| Precision | ~68% |
| Recall | ~75% |
| F1-Score | ~71% |
| ROC-AUC | ~86% |

### Variables más importantes para predecir churn:
1. **Tenure Months** (Antigüedad)
2. **Contract_Month-to-month** (Contrato mensual)
3. **Internet Service_Fiber optic** (Fibra óptica)
4. **Monthly Charges** (Cargos mensuales)
5. **Online Security_No** (Sin seguridad online)
6. **Tech Support_No** (Sin soporte técnico)
7. **Payment Method_Electronic check** (Cheque electrónico)

---

## 💡 Plan de Retención

| Riesgo | Probabilidad | Acción | Descuento | Prioridad |
|--------|-------------|--------|-----------|-----------|
| Bajo | 0-40% | Programa de fidelización estándar | 0% | Baja |
| Moderado | 40-60% | Promoción personalizada | 10-15% | Media |
| Alto | 60-80% | Descuento especial + beneficios | 20% | Alta |
| Crítico | 80-100% | Contacto inmediato + oferta exclusiva | 30% | Urgente |

---

## 🖥️ Dashboard Interactivo (Streamlit)

Ejecutar el dashboard con:
```bash
streamlit run app/app.py
```

Características:
- KPIs de churn en tiempo real
- Predicción individual de churn
- Predicción por lotes
- Comparación visual de modelos
- Plan de retención personalizado
- Visualización de distribución de riesgo
- Descarga de predicciones en CSV

---

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/Ferlola/Telco-Customer-Churn.git
cd Telco-Customer-Churn

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 📓 Ejecución del Proyecto

### Opción 1: Notebooks Jupyter
```bash
jupyter notebook notebooks/
```
Ejecutar en orden:
1. `01_EDA.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_Modeling.ipynb`
4. `04_Evaluation.ipynb`

### Opción 2: Dashboard Streamlit
```bash
streamlit run app/app.py
```

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|-----------|---------|-----------|
| Python | 3.11+ | Lenguaje principal |
| Pandas | 1.5+ | Manipulación de datos |
| NumPy | 1.24+ | Cómputo numérico |
| Matplotlib | 3.7+ | Visualización |
| Seaborn | 0.12+ | Visualización estadística |
| Scikit-learn | 1.2+ | Modelos y preprocesamiento |
| XGBoost | 1.7+ | Gradient Boosting |
| SHAP | 0.41+ | Interpretabilidad |
| Imbalanced-learn | 0.10+ | SMOTE |
| Joblib | 1.2+ | Serialización |
| Streamlit | 1.25+ | Dashboard interactivo |

---

## 📁 Estructura del Código Modular

### `src/preprocessing.py`
- `load_data()` - Carga del dataset
- `clean_data()` - Limpieza y tratamiento de nulos
- `drop_unnecessary_columns()` - Eliminación de columnas no predictivas
- `split_features_target()` - Separación X/y
- `identify_column_types()` - Clasificación de tipos de columna
- `encode_columns()` - Codificación (Label + One-Hot)
- `scale_features()` - Estandarización con StandardScaler
- `apply_smote()` - Balanceo con SMOTE
- `preprocess_data()` - Pipeline completo

### `src/feature_engineering.py`
- `create_avg_monthly_spend()` - Gasto promedio mensual
- `create_tenure_group()` - Grupos de antigüedad
- `create_num_services()` - Cantidad de servicios
- `create_premium_customer()` - Cliente premium
- `create_new_customer()` - Cliente nuevo
- `create_engagement_score()` - Score de engagement
- `engineer_features()` - Pipeline de ingeniería

### `src/train_models.py`
- `train_logistic_regression()` - Logistic Regression
- `train_decision_tree()` - Decision Tree con GridSearchCV
- `train_random_forest()` - Random Forest con GridSearchCV
- `train_xgboost()` - XGBoost con GridSearchCV
- `train_all_models()` - Entrenamiento completo
- `save_model()` - Persistencia del modelo

### `src/evaluate.py`
- `evaluate_model()` - Métricas completas
- `plot_confusion_matrix()` - Matriz de confusión
- `plot_roc_curve()` - Curva ROC
- `plot_precision_recall_curve()` - Curva PR
- `compare_models()` - Tabla comparativa
- `plot_feature_importance()` - Importancia de variables
- `explain_with_shap()` - Análisis SHAP

### `src/retention.py`
- `get_risk_level()` - Clasificación de riesgo
- `classify_churn()` - Predicción binaria
- `predict_single_client()` - Predicción individual
- `predict_batch()` - Predicción por lotes
- `retention_summary()` - Resumen del plan

---

## 🔮 Mejoras Futuras

1. **Deep Learning:** Probar redes neuronales con TensorFlow/PyTorch
2. **Feature Store:** Implementar Feature Store para producción
3. **API REST:** Desplegar modelo como API con FastAPI
4. **MLflow:** Tracking de experimentos
5. **CI/CD:** Pipeline automatizado de entrenamiento
6. **Seguimiento:** Monitorear efectividad de intervenciones
7. **A/B Testing:** Evaluar impacto real del plan de retención
8. **Segmentación avanzada:** Clusterización de clientes para campañas más precisas

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## ✨ Autor

**Científico de Datos Senior**

Proyecto desarrollado como parte de un portafolio profesional de Ciencia de Datos, siguiendo la metodología CRISP-DM y mejores prácticas de MLOps.

---

*"Predecir para retener, analizar para crecer."*
