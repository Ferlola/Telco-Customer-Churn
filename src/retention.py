import pandas as pd
import numpy as np


RETENTION_STRATEGIES = [
    {
        'min_prob': 0.0,
        'max_prob': 0.40,
        'level': 'Bajo',
        'label': 'Bajo Riesgo',
        'action': 'Programa de fidelización estándar',
        'description': 'Mantener al cliente satisfecho con comunicaciones regulares y recordatorios de beneficios.',
        'discount': '0%',
        'priority': 'Baja',
        'contact': 'Email automático trimestral'
    },
    {
        'min_prob': 0.40,
        'max_prob': 0.60,
        'level': 'Medio',
        'label': 'Riesgo Moderado',
        'action': 'Promoción personalizada',
        'description': 'Ofrecer upgrade gratuito por 1 mes o descuento en servicios adicionales.',
        'discount': '10-15%',
        'priority': 'Media',
        'contact': 'Email personalizado + llamada de servicio'
    },
    {
        'min_prob': 0.60,
        'max_prob': 0.80,
        'level': 'Alto',
        'label': 'Alto Riesgo',
        'action': 'Descuento especial + beneficios',
        'description': 'Aplicar descuento del 20% por 3 meses en el plan actual y ofrecer servicio prioritario.',
        'discount': '20%',
        'priority': 'Alta',
        'contact': 'Llamada telefónica de retención + email'
    },
    {
        'min_prob': 0.80,
        'max_prob': 1.01,
        'level': 'Crítico',
        'label': 'Riesgo Crítico',
        'action': 'Contacto inmediato + oferta exclusiva',
        'description': 'Gerente de retención contacta en 24h. Oferta personalizada: descuento del 30% por 6 meses + servicios premium gratis.',
        'discount': '30%',
        'priority': 'Urgente',
        'contact': 'Contacto telefónico inmediato por ejecutivo senior'
    }
]


def get_risk_level(probability):
    for strategy in RETENTION_STRATEGIES:
        if strategy['min_prob'] <= probability < strategy['max_prob']:
            return strategy
    return RETENTION_STRATEGIES[-1]


def classify_churn(probability, threshold=0.5):
    churn_prediction = 'Sí' if probability >= threshold else 'No'
    return churn_prediction


def predict_single_client(model, client_data, threshold=0.5):
    proba = model.predict_proba(client_data)[0, 1]
    classification = classify_churn(proba, threshold)
    risk_strategy = get_risk_level(proba)

    return {
        'Probabilidad': round(proba * 100, 1),
        'Clasificación': classification,
        'Nivel de Riesgo': risk_strategy['label'],
        'Riesgo Level': risk_strategy['level'],
        'Acción Recomendada': risk_strategy['action'],
        'Descripción': risk_strategy['description'],
        'Descuento Sugerido': risk_strategy['discount'],
        'Prioridad': risk_strategy['priority'],
        'Tipo de Contacto': risk_strategy['contact'],
        'Threshold usado': threshold
    }


def predict_batch(model, X, threshold=0.5):
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)

    results = []
    for prob in probas:
        risk_strategy = get_risk_level(prob)
        results.append({
            'Probabilidad': round(prob * 100, 1),
            'Predicción': 'Sí' if prob >= threshold else 'No',
            'Nivel de Riesgo': risk_strategy['label'],
            'Acción Recomendada': risk_strategy['action'],
            'Descuento': risk_strategy['discount'],
            'Prioridad': risk_strategy['priority'],
            'Contacto': risk_strategy['contact']
        })
    return pd.DataFrame(results)


def retention_summary(predictions_df):
    print("\n=== RESUMEN DEL PLAN DE RETENCIÓN ===\n")
    risk_counts = predictions_df['Nivel de Riesgo'].value_counts()
    print("Distribución por nivel de riesgo:")
    for level, count in risk_counts.items():
        pct = count / len(predictions_df) * 100
        print(f"  {level}: {count} clientes ({pct:.1f}%)")

    total_at_risk = len(predictions_df[predictions_df['Predicción'] == 'Sí'])
    print(f"\nTotal clientes en riesgo de abandono: {total_at_risk}")
    print(f"Total clientes analizados: {len(predictions_df)}")

    high_risk = len(predictions_df[predictions_df['Nivel de Riesgo'].isin(['Alto Riesgo', 'Riesgo Crítico'])])
    print(f"Clientes en riesgo Alto o Crítico: {high_risk}")
    print(f"Requieren atención inmediata: {high_risk}")

    if high_risk > 0:
        avg_discount = 20
        estimated_savings = high_risk * 500
        print(f"\nEstimación de impacto:")
        print(f"  Descuento promedio sugerido: {avg_discount}%")
        print(f"  Ahorro estimado (CLTV retenido): ${estimated_savings:,}")

    return risk_counts.to_dict()
