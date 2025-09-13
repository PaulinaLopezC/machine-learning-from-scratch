# machine-learning-from-scratch
Módulo 2 - Implementación de una técnica de aprendizaje máquina. (Portafolio Análisis e Implementación)

Evidencia para la materia "Inteligencia artificial avanzada para la ciencia de datos I (TC3006C)"
#### Guadalupe Paulina López Cuevas | A01701095 - Análisis y desarrollo de modelos predictivos para datos biomédicos

# Implementación de Aprendizaje Máquina 
Predictor de Severidad de Parkinson mediante Análisis de Voz
Este proyecto implementa y compara modelos de machine learning para predecir la severidad de la enfermedad de Parkinson (usando las escalas motor_UPDRS y total_UPDRS) a partir de mediciones biomédicas de la voz. El dataset utilizado contiene grabaciones de voz de 42 pacientes en etapas tempranas de Parkinson, con 22 atributos acústicos iniciales que incluyen métricas de jitter, shimmer, ruido y características no lineales.

## Contenido
- REPORTE -> Evidencia1_PaulinaLopezCuevas_A01701095.pdf
- Modelo SIN framework -> ml_algorithm.py
- Modelo CON framework -> ml_framework.py
- Dataset -> parkinsons_updrs.data
- Dataset info -> parkinsons_updrs.names
- Gráficas Pred vs Real SIN framework -> Algorithm_PredvsReal_Train&Val&Test.png
- Gráficas Pred vs Real CON framework -> Framework_PredvsReal_Train&Val&Test.png
- Gráficas MSE y Accuracy -> RandomForest_MSE&Acc_Graphs.png

## Objetivos
- Realizar un proceso completo de ETL y limpieza de datos biomédicos.
- Analizar correlaciones y multicolinealidad entre variables.
- Implementar y evaluar modelos de regresión (lineal, polinomial y Random Forest).
- Optimizar hiperparámetros para lograr el mejor equilibrio entre sesgo y varianza.
- Comparar el desempeño predictivo de diferentes enfoques de modelado.

## Dataset
Nombre: Parkinson's Telemonitoring

Fuente: Universidad de Oxford

Muestras: 5,875 grabaciones de voz

Pacientes: 42 individuos

Características: 22 atributos acústicos y clínicos

Target: total_UPDRS (escalas de severidad)

## Implementación Técnica
Preprocesamiento y Limpieza

Eliminación de valores nulos y duplicados

Remoción de outliers mediante percentiles (25%-75%)

Eliminación de variables no relevantes (subject#, test_time)

Shuffle de datos para evitar sesgos en el entrenamiento

Reducción de features basada en análisis de correlación

Escalamiento mediante StandardScaler

## Modelos Implementados
### Regresión Lineal Manual
- Implementación desde cero con Gradiente Descendente
- Función de costo MSE y optimización iterativa
- Regresión Polinomial (2do grado)
- Transformación de features a términos polinomiales
- Mejora significativa respecto al modelo lineal

### Random Forest Regressor
Implementado con Scikit-Learn

Ajuste fino de hiperparámetros de regularización

Búsqueda del equilibrio óptimo sesgo-varianza

## Métricas de Evaluación
- Error Cuadrático Medio (MSE)
- Coeficiente de Determinación (R²)
- Análisis de sesgo y varianza
- Validación en conjuntos de train/validation/test (60/20/20)

## Resultados Clave
### Regresión Polinomial
- MSE train: 44.52 / MSE validation: 73.14
- R² train: 0.248 / R² validation: 0.235
- Diagnóstico: High bias (underfitting)

### Random Forest Optimizado
- MSE train: 3.38 / MSE test: 10.70
- R² train: 0.970 / R² test: 0.906
- Diagnóstico: Optimal fit (low bias, low variance)

## Conclusiones
El proyecto demostró que los modelos de regresión lineal y polinomial presentan limitaciones significativas para modelar la complejidad de datos biomédicos del Parkinson, resultando en alto sesgo (underfitting) debido a las bajas correlaciones lineales. La implementación de Random Forest con ajuste fino de hiperparámetros superó estas limitaciones, alcanzando un equilibrio óptimo entre sesgo y varianza con excelente capacidad de generalización (R² > 0.90). Los resultados validan la superioridad de los modelos basados en árboles para capturar patrones no lineales en problemas complejos del mundo real.

## Requisitos Técnicos
- Python 3.8+
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
