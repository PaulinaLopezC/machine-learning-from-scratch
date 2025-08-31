# ---------------------------------------------------------------------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
# Guadalupe Paulina López Cuevas
# A01701095
# 
# Linear Regression Model + Logaritmic Cost Function
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math

# ==================================================================================
# LIMPIEZA DE DATOS
# ==================================================================================
# Funcion para cargar el data set desde un archivo para manipular los datos
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    print("LOADED DATASET\n", df)
    return df

# Funcion para revisar porcentaje de valores nulos en el nuestro dataset
def null_count(df):
    df_nulos = df.isnull().mean()*100
    print("NULL DATA\n", df_nulos)
    return df_nulos

# Funcion para revisar si existe algun valor duplicado y quitarlo del dataset
def remove_duplicate(df):
  df_nonduplicate = df.drop_duplicates()
  print("NO MORE DUPLICATED\n", df_nonduplicate)
  return df_nonduplicate

# Funcion para quitar alguna colunma en especifico
# Objetivo: remover datos del dataset que no sean considerados relevantes para nuestro entrenamiento
def remove_subject_column(df, column):
    return df.drop(column, axis=1)

# Funcion para deshacernos de los valores que son anomalias(ruido o errores de medicion)
# Objetivo: Tomar los valores que se encuentren entre el 25% y 75% de los datos originales
# #para despreciar extremos muy fuera del comportamiento normal
def outliers_drop(df, columns):
  for i in columns:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]
  return df

# Funcion que nos sirve para remover todos los datos nulos encontrados en el dataset, de ser necesario
def remove_nulls(df):
    df_clean = df.dropna()
    return df_clean

# ==================================================================================
# GRAFICAS PARA SABER CORRELACION ENTRE FEATURES Y TARGETS
# ==================================================================================
# Funcion para sacar la matriz de correlacion de un dataset
def correlation_matrix(df):
    print("MATRIZ DE CORRELACIÓN\n")
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", center=0)
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.show()

# Funcion que saca la grafica de pares, para ver si se comportar de manera lineal
def pair_graphic(df):
    sns.pairplot(df, height=1.5)
    plt.show()

# ==================================================================================
# TRANSFORMACION A TERMINOS POLINOMIALES
# ==================================================================================
# Funcion para crear variables polinomiales
def create_polynomial(df, y_target):
    # Separar las variables independientes
    X = df.drop(y_target, axis=1)
    # Crear el polinomio de variables de 2do grado
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    # Guardar el polinomio
    df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
    #regresar el target al dataset
    df_poly[y_target] = df[y_target]
    return df_poly

# ==================================================================================
# DIVISION DEL DATASET EN TRAIN Y TEST
# ==================================================================================
# Funcion para dividir y deplegar shape del dataset de train y test
def divide_train_test(df, y_target):
    x = df.drop(y_target, axis=1)
    y = df[y_target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Shape X_train:", x_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape X_test:", x_test.shape)
    print("Shape y_test:", y_test.shape)
    return x_train, y_train, x_test, y_test

# ==================================================================================
# ESCALAMIENTO DE VARIABLES INDEPENDIENTES Y REDIMENSION DE VARIABLES DEPENDIENTES
# ==================================================================================
# Funcion para hacer transformacion de variables dependientes e independientes
def trans_dep_indep(x_train, y_train, x_test, y_test):
    df_y_train = y_train.to_numpy()
    df_y_test = y_test.to_numpy()
    scaler = StandardScaler()
    df_x_train = scaler.fit_transform(x_train)
    df_x_test = scaler.transform(x_test)
    return df_x_train, df_y_train, df_x_test, df_y_test

# ==================================================================================
# REGRESION LINEAL
# ==================================================================================
# Funcion para calcular la prediccion del modelo
def hyp_theta(x, theta, b):
  y = 0
  for i in range(len(theta)):
    y += x[i] * theta[i]
  return (y + b)

# Funcion para sacar el error cuadratico medio prediccion vs valor real
def MSE(data, theta, b, Y):
  cost = 0
  m = len(data)
  for i in range(m):
    hyp = hyp_theta(data[i], theta, b)
    cost += (hyp - Y[i])**2
  return (cost / (2*m))

# Funcion para actualizar los valores de los parametros (durante el entrenamiento)
def update(data, theta, b, Y, alfa):
  theta_new = np.zeros(len(theta))
  m = len(data)
  n = len(theta)
  for j in range(n):
    grad = 0
    for i in range(m):
      error = hyp_theta(data[i], theta, b) - Y[i]
      grad += error * data[i][j]
    theta_new[j] = theta[j] - alfa/m * grad

  grad_b = 0
  for i in range(m):
    grad_b += (hyp_theta(data[i], theta, b) - Y[i])
  b_new = b - alfa/m * grad_b
  return theta_new, b_new

# ==================================================================================
# GRAFICAS PARA VISUALIZAR ERROR, PREDICCION VS VALOR REAL (TRAIN Y TEST)
# ==================================================================================
# Funcion para graficar error de entrenamiento, comparacion valor de prediccion vs real de train y test
def plot_results(error_train, df_y_train, df_y_pred_train, r2_train, df_y_test, df_y_pred_test, r2_test)  :
    plt.figure(figsize=(15, 5))

    # Evolución del error
    plt.subplot(1, 3, 1)
    sns.lineplot(x=range(len(error_train)), y=error_train)
    plt.title('Evolución del Error (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(True)

    # Predicciones vs Valores reales (train)
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=df_y_train, y=df_y_pred_train, alpha=0.5, color='blue')
    plt.plot([df_y_train.min(), df_y_train.max()], [df_y_train.min(), df_y_train.max()], 'r--', lw=2)
    plt.title(f'Train: R² = {r2_train:.3f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)

    # Predicciones vs Valores reales (test)
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=df_y_test, y=df_y_pred_test, alpha=0.5, color='purple')
    plt.plot([df_y_test.min(), df_y_test.max()], [df_y_test.min(), df_y_test.max()], 'r--', lw=2)
    plt.title(f'Test: R² = {r2_test:.3f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones Escaladas')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    df = load_data('parkinsons_updrs.data')
    # Desplegar informacion sobre cada una de las columnas(tipo de datos)
    df.info()

    # ==================================================================================
    # LIMPIEZA DE DATOS
    # ==================================================================================
    # Contar cuantos valores nulos hay en cada columna
    null_count(df)
    # Remover los datos nulos si es que hay alguno
    df = remove_nulls(df)
    # Remover los valores duplicados
    df_clean = remove_duplicate(df)
    # Desplegar descripcion sobre
    print("INFORMACION SOBRE DATASET\n", df_clean.describe())
    # Quitar columna 'subject#'
    df_clean = remove_subject_column(df_clean, 'subject#')
    # Quitar columna 'test_time'
    df_clean = remove_subject_column(df_clean, 'test_time')
    # Crear columns para poner todas las columnas a las que queremos quitar los valores que pueden ser anomalias
    columns =  ['Jitter(%)', 'Jitter(Abs)', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    df_clean = outliers_drop(df_clean, columns)
    # Revolver el dataset para tener los datos desordenados
    df_shuffle = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    print("DATASET SHUFFLED\n", df_shuffle)

    # ==================================================================================
    # GRAFICOS DE CORRELACION LINEAL DE DATASET
    # ==================================================================================
    # Sacar matriz de correlacioin
    correlation_matrix(df_shuffle)
    # Sacar plot de features y targest
    pair_graphic(df_shuffle)

    # ==================================================================================
    # REDUCCION DE FUNCION, GRAFICOS DE CORRELACION LINEAL DE DATASET REDUCIDO
    # ==================================================================================
    df_reduce = df_shuffle[['age', 'sex', 'total_UPDRS', 'Jitter(%)','Shimmer:APQ11','NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
    correlation_matrix(df_reduce)
    pair_graphic(df_reduce)

    # ==================================================================================
    # TRANFORMACION A TERMINOS POLINOMIALES
    # ==================================================================================
    # Crear caracteristicas polinomiales
    y_target = 'total_UPDRS'
    df_final = create_polynomial(df_reduce, y_target)
    print("DATASET POLYNOMIAL\n", df_final)

    # ==================================================================================
    # DIVISION DEL DATASET A TRAIN Y TEST, ESCALAMIENTO Y REDIMENSION
    # ==================================================================================
    # Dividir el dataset en train y test
    x_train, y_train, x_test, y_test = divide_train_test(df_final, y_target)
    
    # Transformacion para variables dependientes e independientes
    df_x_train, df_y_train, df_x_test, df_y_test = trans_dep_indep(x_train, y_train, x_test, y_test)

    # ==================================================================================
    # INICIALIZACION DE PARAMETROS PARA EL ENTRENAMIENTO DEL MODELO
    # ==================================================================================
    print("INICIALIZAR PARAMETROS")
    # Weights
    theta_w = np.zeros(df_x_train.shape[1])
    print("Weights = ", theta_w)
    # Error
    error_train = []
    # Bias
    b = 100
    print("Bias = ", b)
    # Learning rate
    alfa = 0.01
    print("learning rate = ", alfa)
    # Epocas
    epoch = 3000
    print("Epochs = ", epoch)

    # ==================================================================================
    # ENTRENAMIENTO DEL MODELO
    # ==================================================================================
    # Entrenamiento del modelo
    for i in range(epoch) : # Increased epochs
        current_error = MSE(df_x_train, theta_w, b, df_y_train)
        error_train.append(current_error)
        # Calculate R-squared
        df_y_pred_train = [hyp_theta(x, theta_w, b) for x in df_x_train]
        r2_train = r2_score(df_y_train, df_y_pred_train)
        if (i + 1) % 100 == 0:
            print(f"Epoch {i+1}: MSE = {current_error:.6f}, R-squared = {r2_train:.6f}")

        if current_error < 0.001:
            break
        theta_w, b = update(df_x_train, theta_w, b, df_y_train, alfa)

    # ==================================================================================
    # IMPRIMIR RESULTADOS DEL ENTRENAMIENTO
    # ==================================================================================
    print(f"\nResultados finales después de {i+1} épocas:")
    print(f"Error final: {error_train[-1]:.6f}") # Access the single value in the array
    print(f"Theta final: {theta_w}")
    print(f"b final: {b:.2f}")

    print(f"\nPredicciones vs Valores reales (entrenamiento):")
    for i in range(5): # Print for first 5 samples
        pred = hyp_theta(df_x_train[i], theta_w, b)
        print(f"-> Pred: {pred:.4f}, Real: {df_y_train[i]}")

    # ==================================================================================
    # PRUEBA DEL MODELO
    # ==================================================================================
    # Calculate MSE on the test set
    test_error = MSE(df_x_test, theta_w, b, df_y_test)
    df_y_pred_test = [hyp_theta(x, theta_w, b) for x in df_x_test]
    r2_test = r2_score(df_y_test, df_y_pred_test)
    print(f"Epoch {i+1}: MSE = {test_error:.6f}, R-squared = {r2_test:.6f}")
    print(f"\nError Cuadrático Medio (MSE) en el conjunto de prueba: {test_error:.6f}")

    # ==================================================================================
    # GRAFICAS PARA VISUALIZAR ERROR, PREDICCION VS VALOR REAL (TRAIN Y TEST) 
    # ==================================================================================
    # Graficas
    plot_results(error_train, df_y_train, df_y_pred_train, r2_train, df_y_test, df_y_pred_test, r2_test)    

if __name__ == "__main__":
    main()