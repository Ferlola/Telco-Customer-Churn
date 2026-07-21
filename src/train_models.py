import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train, random_state=42):
    print("\n=== ENTRENANDO REGRESIÓN LOGÍSTICA ===")
    lr = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced'
    )
    lr.fit(X_train, y_train)

    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return lr


def train_decision_tree(X_train, y_train, random_state=42):
    print("\n=== ENTRENANDO ÁRBOL DE DECISIÓN ===")
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    dt = DecisionTreeClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        dt, param_grid, n_iter=30, cv=5,
        scoring='roc_auc', n_jobs=-1, verbose=0, random_state=random_state
    )
    search.fit(X_train, y_train)

    print(f"Mejores parámetros: {search.best_params_}")
    print(f"Mejor CV ROC-AUC: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def train_random_forest(X_train, y_train, random_state=42):
    print("\n=== ENTRENANDO RANDOM FOREST ===")
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=30, cv=5,
        scoring='roc_auc', n_jobs=-1, verbose=0, random_state=random_state
    )
    search.fit(X_train, y_train)

    print(f"Mejores parámetros: {search.best_params_}")
    print(f"Mejor CV ROC-AUC: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def train_xgboost(X_train, y_train, random_state=42):
    print("\n=== ENTRENANDO XGBOOST ===")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    xgb = XGBClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=30, cv=5,
        scoring='roc_auc', n_jobs=-1, verbose=0, random_state=random_state
    )
    search.fit(X_train, y_train)

    print(f"Mejores parámetros: {search.best_params_}")
    print(f"Mejor CV ROC-AUC: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def save_model(model, filename, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")
    return filepath


def train_all_models(X_train, y_train, save=True):
    models = {}

    lr = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr
    if save:
        save_model(lr, 'logistic_regression.pkl')

    dt, dt_params = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt
    models['Decision Tree Params'] = dt_params
    if save:
        save_model(dt, 'decision_tree.pkl')

    rf, rf_params = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf
    models['Random Forest Params'] = rf_params
    if save:
        save_model(rf, 'random_forest.pkl')

    xgb, xgb_params = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb
    models['XGBoost Params'] = xgb_params
    if save:
        save_model(xgb, 'xgboost.pkl')

    return models


def load_model(filename, model_dir='models'):
    filepath = os.path.join(model_dir, filename)
    model = joblib.load(filepath)
    print(f"Modelo cargado desde: {filepath}")
    return model
