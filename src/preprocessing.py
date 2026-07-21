import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import engineer_features


def load_data(filepath='data/raw/Telco_customer_churn.xlsx'):
    df = pd.read_excel(filepath)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df):
    df_clean = df.copy()

    # Convertir Total Charges a numérico
    df_clean['Total Charges'] = pd.to_numeric(df_clean['Total Charges'], errors='coerce')

    # Imputar valores faltantes de Total Charges con la mediana
    median_charges = df_clean['Total Charges'].median()
    df_clean['Total Charges'] = df_clean['Total Charges'].fillna(median_charges)

    # Imputar Churn Reason con 'Unknown' para clientes que no abandonaron
    df_clean['Churn Reason'] = df_clean['Churn Reason'].fillna('Unknown')

    return df_clean


def drop_unnecessary_columns(df):
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City',
        'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
        'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
    ]
    df_dropped = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df_dropped


def split_features_target(df, target_col='Churn Value'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def identify_column_types(X):
    binary_cols = []
    multi_cat_cols = []
    numeric_cols = []

    for col in X.columns:
        if pd.api.types.is_categorical_dtype(X[col].dtype):
            multi_cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col].dtype):
            if X[col].nunique() <= 2:
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(X[col].dtype):
            if X[col].nunique() <= 2:
                binary_cols.append(col)
            else:
                multi_cat_cols.append(col)
        else:
            multi_cat_cols.append(col)

    return binary_cols, multi_cat_cols, numeric_cols


def encode_columns(X, binary_cols, multi_cat_cols):
    X_encoded = X.copy()
    encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    X_encoded = pd.get_dummies(X_encoded, columns=multi_cat_cols, drop_first=True, dtype=int)

    return X_encoded, encoders


def scale_features(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    if isinstance(X_train, pd.DataFrame):
        nan_cols = X_train.columns[X_train.isnull().any()].tolist()
        inf_cols = X_train.columns[np.isinf(X_train.select_dtypes(include=[np.number])).any()].tolist()
        if nan_cols or inf_cols:
            print(f"Columnas con NaN: {nan_cols}" if nan_cols else "", end=" ")
            print(f"Columnas con Inf: {inf_cols}" if inf_cols else "")
            X_train = X_train.fillna(X_train.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan)
            X_train = X_train.fillna(0)

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"\nSMOTE aplicado. Shape original X_train: {X_train.shape}")
    print(f"Shape después de SMOTE X_train: {X_resampled.shape}")
    print(f"Distribución después de SMOTE:\n{y_resampled.value_counts()}")
    return X_resampled, y_resampled


def get_preprocessor_pipeline():
    return {
        'load': load_data,
        'clean': clean_data,
        'drop': drop_unnecessary_columns,
        'split': split_features_target,
        'identify': identify_column_types,
        'encode': encode_columns,
        'scale': scale_features,
        'smote': apply_smote
    }


def preprocess_data(df=None, filepath='data/raw/Telco_customer_churn.xlsx',
                    test_size=0.2, random_state=42, apply_smote_flag=True,
                    return_scaler=True, return_encoders=True,
                    apply_feature_engineering=True):
    if df is None:
        df = load_data(filepath)

    df = clean_data(df)
    df = drop_unnecessary_columns(df)

    if apply_feature_engineering:
        df = engineer_features(df)

    X, y = split_features_target(df, 'Churn Value')
    binary_cols, multi_cat_cols, numeric_cols = identify_column_types(X)

    X_encoded, encoders = encode_columns(X, binary_cols, multi_cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_features = [c for c in numeric_cols if c in X_train.columns]
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_features)

    if apply_smote_flag:
        X_train_res, y_train_res = apply_smote(X_train_scaled, y_train, random_state)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    result = {
        'X_train': X_train_res,
        'X_test': X_test_scaled,
        'y_train': y_train_res,
        'y_test': y_test,
        'feature_names': list(X_encoded.columns),
        'scaler': scaler,
        'encoders': encoders,
        'binary_cols': binary_cols,
        'multi_cat_cols': multi_cat_cols,
        'numeric_cols': numeric_cols,
        'df_clean': df
    }

    return result


def save_preprocessing_artifacts(prep_result, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(prep_result['scaler'], os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(prep_result['encoders'], os.path.join(output_dir, 'encoders.pkl'))
    print(f"Artefactos guardados en {output_dir}/")
