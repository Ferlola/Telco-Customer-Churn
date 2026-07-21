import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.retention import predict_single_client, predict_batch, get_risk_level, RETENTION_STRATEGIES
from src.evaluate import evaluate_model

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1a5276; font-weight: bold;}
    .risk-low {color: #27ae60; font-weight: bold;}
    .risk-moderate {color: #f39c12; font-weight: bold;}
    .risk-high {color: #e67e22; font-weight: bold;}
    .risk-critical {color: #e74c3c; font-weight: bold;}
    .metric-card {background-color: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)


def main():
    @st.cache_resource
    def load_artifacts():
        try:
            model = joblib.load(os.path.join(BASE_DIR, 'models/xgboost.pkl'))
            scaler = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))
            encoders = joblib.load(os.path.join(BASE_DIR, 'models/encoders.pkl'))
            prep_data = joblib.load(os.path.join(BASE_DIR, 'data/processed/preprocessed_data.pkl'))
            return model, scaler, encoders, prep_data
        except Exception as e:
            st.error(f"Error cargando artefactos: {e}")
            return None, None, None, None

    @st.cache_data
    def load_data():
        df = pd.read_excel(os.path.join(BASE_DIR, 'data/raw/Telco_customer_churn.xlsx'))
        return df

    model, scaler, encoders, prep_data = load_artifacts()
    df_raw = load_data()

    st.markdown('<p class="main-header">📊 Telco Customer Churn Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("### Sistema Inteligente de Predicción de Abandono y Plan de Retención")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 KPIs y Distribución",
        "🔍 Predicción Individual",
        "📋 Predicción por Lotes",
        "📊 Comparación de Modelos",
        "💡 Plan de Retención"
    ])

    # ============ TAB 1: KPIs ============
    with tab1:
        st.header("KPIs de Churn")

        col1, col2, col3, col4 = st.columns(4)
        total = len(df_raw)
        churn_rate = df_raw['Churn Value'].mean() * 100
        churn_count = df_raw['Churn Value'].sum()
        avg_tenure = df_raw['Tenure Months'].mean()
        avg_monthly = df_raw['Monthly Charges'].mean()

        col1.metric("Total Clientes", f"{total:,}")
        col2.metric("Tasa de Churn", f"{churn_rate:.1f}%",
                     delta=f"{churn_count:,} clientes")
        col3.metric("Antigüedad Promedio", f"{avg_tenure:.1f} meses")
        col4.metric("Cargo Mensual Prom.", f"${avg_monthly:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución de Churn")
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#e74c3c']
            df_raw['Churn Label'].value_counts().plot(
                kind='pie', autopct='%1.1f%%', colors=colors,
                labels=['No Churn', 'Churn'], ax=ax, explode=(0, 0.05),
                startangle=90, shadow=True
            )
            ax.set_ylabel('')
            ax.set_title('')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Churn por Tipo de Contrato")
            fig, ax = plt.subplots(figsize=(8, 6))
            cross = pd.crosstab(df_raw['Contract'], df_raw['Churn Label'],
                                normalize='index') * 100
            cross.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
            ax.set_title('Tasa de Churn por Contrato')
            ax.set_xlabel('Contrato')
            ax.set_ylabel('Porcentaje (%)')
            ax.legend(['No Churn', 'Churn'])
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)
            plt.close()

        st.subheader("Variables más importantes")
        if prep_data:
            feature_names = prep_data['feature_names']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(15)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feat_imp, y='Feature', x='Importance',
                           palette='viridis', ax=ax)
                ax.set_title('Top 15 Variables Predictoras')
                st.pyplot(fig)
                plt.close()

    # ============ TAB 2: Predicción Individual ============
    with tab2:
        st.header("🔍 Predicción Individual de Churn")
        st.markdown("Ingrese los datos del cliente para predecir su riesgo de abandono.")

        if model is not None and prep_data is not None:
            feature_names = prep_data['feature_names']

            col1, col2, col3 = st.columns(3)

            with col1:
                tenure = st.slider("Tenure (meses)", 1, 72, 12)
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
                total = st.number_input("Total Charges ($)", 18.0, 9000.0, 800.0)
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])

            with col2:
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                phone = st.selectbox("Phone Service", ["No", "Yes"])
                multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No", "Cable"])

            with col3:
                online_sec = st.selectbox("Online Security",
                                          ["No", "Yes", "No internet service"])
                online_back = st.selectbox("Online Backup",
                                           ["No", "Yes", "No internet service"])
                device_prot = st.selectbox("Device Protection",
                                           ["No", "Yes", "No internet service"])
                tech_sup = st.selectbox("Tech Support",
                                        ["No", "Yes", "No internet service"])

            col4, col5 = st.columns(2)

            with col4:
                streaming_tv = st.selectbox("Streaming TV",
                                            ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies",
                                                ["No", "Yes", "No internet service"])

            with col5:
                contract = st.selectbox("Contract",
                                        ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment = st.selectbox("Payment Method",
                                       ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"])

            if st.button("🔮 Predecir Churn", type="primary", width='stretch'):
                input_data = pd.DataFrame([{
                    'Gender': gender, 'Senior Citizen': senior, 'Partner': partner,
                    'Dependents': dependents, 'Tenure Months': tenure,
                    'Phone Service': phone, 'Multiple Lines': multi_lines,
                    'Internet Service': internet, 'Online Security': online_sec,
                    'Online Backup': online_back, 'Device Protection': device_prot,
                    'Tech Support': tech_sup, 'Streaming TV': streaming_tv,
                    'Streaming Movies': streaming_movies, 'Contract': contract,
                    'Paperless Billing': paperless, 'Payment Method': payment,
                    'Monthly Charges': monthly, 'Total Charges': total
                }])

                # Aplicar misma transformación que en training
                input_processed = input_data.copy()

                numeric_cols_in_data = ['Tenure Months', 'Monthly Charges', 'Total Charges']

                encoded_inputs = []
                for col in feature_names:
                    if col in input_processed.columns:
                        encoded_inputs.append(input_processed[col].values[0])
                    elif col in prep_data.get('binary_cols', []):
                        le = encoders.get(col)
                        if le:
                            val = input_processed[col].values[0] if col in input_processed.columns else 'No'
                            try:
                                encoded_inputs.append(le.transform([val])[0])
                            except:
                                encoded_inputs.append(0)
                        else:
                            encoded_inputs.append(0)
                    elif col.startswith(tuple(prep_data.get('multi_cat_cols', []))):
                        base_col = col.split('_')[0]
                        if base_col in input_processed.columns:
                            val = input_processed[base_col].values[0]
                            expected_val = col[len(base_col)+1:]
                            encoded_inputs.append(1 if val == expected_val else 0)
                        else:
                            encoded_inputs.append(0)
                    elif col in numeric_cols_in_data:
                        encoded_inputs.append(input_processed[col].values[0])
                    else:
                        encoded_inputs.append(0)

                input_df = pd.DataFrame([encoded_inputs], columns=feature_names)

                numeric_features = [c for c in prep_data.get('numeric_cols', []) if c in feature_names]
                if numeric_features:
                    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

                st.markdown("---")
                result = predict_single_client(model, input_df)

                col1, col2, col3 = st.columns(3)
                risk_color = {
                    'Bajo Riesgo': '#27ae60',
                    'Riesgo Moderado': '#f39c12',
                    'Alto Riesgo': '#e67e22',
                    'Riesgo Crítico': '#e74c3c'
                }

                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <h3>Probabilidad de Churn</h3>
                        <p style="font-size:3rem; font-weight:bold; color:{risk_color.get(result['Nivel de Riesgo'], '#333')}">
                            {result['Probabilidad']}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    level_color = risk_color.get(result['Nivel de Riesgo'], '#333')
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <h3>Nivel de Riesgo</h3>
                        <p style="font-size:2rem; font-weight:bold; color:{level_color}">
                            {result['Nivel de Riesgo']}
                        </p>
                        <p style="font-size:1.2rem;">Clasificación: <b>{result['Clasificación']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <h3>Acción Recomendada</h3>
                        <p style="font-size:1.1rem;"><b>{result['Acción Recomendada']}</b></p>
                        <p>{result['Descripción']}</p>
                        <p><b>Descuento:</b> {result['Descuento Sugerido']} | <b>Prioridad:</b> {result['Prioridad']}</p>
                        <p><b>Contacto:</b> {result['Tipo de Contacto']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Los modelos no están disponibles. Ejecute primero los notebooks de modelado.")

    # ============ TAB 3: Predicción por Lotes ============
    with tab3:
        st.header("📋 Predicción por Lotes")
        st.markdown("Sube un archivo CSV con datos de clientes para obtener predicciones masivas.")

        uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Archivo cargado: {len(batch_df)} registros")
                st.dataframe(batch_df.head())

                if st.button("Predecir Lote"):
                    st.info("Funcionalidad de predicción por lote requiere preprocesamiento del archivo subido.")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

        st.markdown("### Vista previa de predicciones (datos de test)")
        if prep_data:
            predictions = predict_batch(model, prep_data['X_test'])
            st.dataframe(predictions.head(20), width='stretch')

            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Predicciones (CSV)",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv'
            )

    # ============ TAB 4: Comparación de Modelos ============
    with tab4:
        st.header("📊 Comparación de Modelos")
        metrics_path = os.path.join(BASE_DIR, 'reports/metrics/model_comparison.csv')
        if os.path.exists(metrics_path):
            comparison_df = pd.read_csv(metrics_path)
            st.dataframe(comparison_df, width='stretch')

            fig, ax = plt.subplots(figsize=(10, 6))
            df_melted = comparison_df.melt(id_vars=['Modelo'],
                                           var_name='Métrica', value_name='Valor')
            sns.barplot(data=df_melted, x='Métrica', y='Valor',
                       hue='Modelo', ax=ax)
            ax.set_title('Comparación de Métricas entre Modelos')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Ejecute los notebooks primero para generar las métricas comparativas.")

        col1, col2 = st.columns(2)
        with col1:
            roc_path = os.path.join(BASE_DIR, 'reports/figures/roc_curves.png')
            if os.path.exists(roc_path):
                st.image(roc_path, caption='Curvas ROC', width='stretch')

        with col2:
            pr_path = os.path.join(BASE_DIR, 'reports/figures/pr_curves.png')
            if os.path.exists(pr_path):
                st.image(pr_path, caption='Curvas Precision-Recall', width='stretch')

    # ============ TAB 5: Plan de Retención ============
    with tab5:
        st.header("💡 Plan Inteligente de Retención")

        st.markdown("""
        Basado en la probabilidad de abandono calculada por el modelo, se asignan las siguientes estrategias:
        """)

        strategies_df = pd.DataFrame(RETENTION_STRATEGIES)
        strategies_df['Rango'] = strategies_df.apply(
            lambda r: f"{int(r['min_prob']*100)}% - {int(r['max_prob']*100)}%", axis=1
        )
        display_df = strategies_df[['Rango', 'level', 'label', 'action', 'discount', 'priority', 'contact']]
        display_df.columns = ['Rango de Probabilidad', 'Nivel', 'Etiqueta', 'Acción',
                              'Descuento', 'Prioridad', 'Contacto']
        st.dataframe(display_df, width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Distribución de Riesgo en Clientes Actuales")

        if model is not None and prep_data is not None:
            predictions = predict_batch(model, prep_data['X_test'])
            risk_counts = predictions['Nivel de Riesgo'].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors_risk = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
                risk_counts.plot(kind='bar', ax=ax, color=colors_risk[:len(risk_counts)],
                               edgecolor='black')
                ax.set_title('Distribución por Nivel de Riesgo')
                ax.set_xlabel('Nivel')
                ax.set_ylabel('Clientes')
                ax.tick_params(axis='x', rotation=0)
                for i, v in enumerate(risk_counts.values):
                    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                risk_counts.plot(kind='pie', autopct='%1.1f%%',
                               colors=colors_risk[:len(risk_counts)],
                               ax=ax, startangle=90)
                ax.set_ylabel('')
                ax.set_title('Proporción por Nivel de Riesgo')
                st.pyplot(fig)
                plt.close()

            high_risk = len(predictions[predictions['Nivel de Riesgo'].isin(
                ['Alto Riesgo', 'Riesgo Crítico'])])
            total_pred = len(predictions)
            churn_pred = len(predictions[predictions['Predicción'] == 'Sí'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes Analizados", total_pred)
            col2.metric("Posible Churn", churn_pred,
                       delta=f"{churn_pred/total_pred*100:.1f}%")
            col3.metric("Atención Prioritaria", high_risk,
                       delta=f"{high_risk/total_pred*100:.1f}%")

            st.subheader("Recomendaciones por Cliente")
            st.dataframe(predictions.head(20), width='stretch')


if __name__ == '__main__':
    main()
