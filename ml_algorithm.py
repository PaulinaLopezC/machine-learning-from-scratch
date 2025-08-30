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
import math

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

# Funcion para hacer transfornacion de variables dependientes e independientes
def trans_dep_indep(x_train, y_train, x_test, y_test):
    df_y_train = y_train.to_numpy()
    df_y_test = y_test.to_numpy()
    scaler = StandardScaler()
    df_x_train = scaler.fit_transform(x_train)
    df_x_test = scaler.transform(x_test)
    return df_x_train, df_y_train, df_x_test, df_y_test

def hyp_theta(x, theta, b):
  y = 0
  for i in range(len(theta)):
    y = y + x[i] * theta[i]
  y = y + b
  return y

def MSE(data, theta, b, Y):
  cost = 0
  m = len(data)
  for i in range(m):
    hyp = hyp_theta(data[i], theta, b)
    cost = cost + (hyp - Y[i])**2
  mean_cost = cost/(2*m)
  return mean_cost

def update(data, theta, b, Y, alfa):
  theta_new = np.zeros(len(theta))
  m = len(data)
  n = len(theta)
  for j in range(n):
    grad = 0
    for i in range(m):
      error = hyp_theta(data[i], theta, b) - Y[i]
      if j < len(data[i]):
        grad = grad + error * data[i][j]
      else:
        grad = grad + error * 0
    theta_new[j] = theta[j] - alfa/m * grad

  grad = 0
  for i in range(m):
    grad = grad + (hyp_theta(data[i], theta, b) - Y[i])
  b_new = b - alfa/m * grad
  return theta_new, b_new

def main():
    df = load_data('parkinsons_updrs.data')
    # Desplegar informacion sobre cada una de las columnas(tipo de datos)
    df.info()

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

    # Sacar matriz de correlacioin
    #correlation_matrix(df_shuffle)
    # Sacar plot de features y targest
    #pair_graphic(df_shuffle)

    # REDUCCION DE FUNCION
    df_reduce = df_shuffle[['age', 'sex', 'total_UPDRS', 'Jitter(%)','Shimmer:APQ11','NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
    #correlation_matrix(df_reduce)
    #pair_graphic(df_reduce)

    # Crear caracteristicas polinomiales
    y_target = 'total_UPDRS'
    df_final = create_polynomial(df_reduce, y_target)
    print("DATASET POLYNOMIAL\n", df_final)

    # Dividir el dataset en train y test
    x_train, y_train, x_test, y_test = divide_train_test(df_final, y_target)
    
    # Transformacion para variables dependientes e independientes
    df_x_train, df_y_train, df_x_test, df_y_test = trans_dep_indep(x_train, y_train, x_test, y_test)

    print("INICIALIZAR PARAMETROS")
    # Weights
    theta_w = np.zeros(df_x_train.shape[1])
    print("Weights = ", theta_w)
    # Error
    error = []
    # Bias
    b = 100
    print("Bias = ", b)
    # Learning rate
    alfa = 0.01
    print("learning rate = ", alfa)
    # Epocas
    epoch = 400
    print("Epochs = ", epoch)

    # Entrenamiento del modelo
    for i in range(epoch) : # Increased epochs
        current_error = MSE(df_x_train, theta_w, b, df_y_train)
        error.append(current_error)
        if current_error < 0.001:
            break
        theta_w, b = update(df_x_train, theta_w, b, df_y_train, alfa)

    print(f"\nResultados finales después de {i+1} épocas:")
    print(f"Error final: {error[-1]:.6f}") # Access the single value in the array
    print(f"Theta final: {theta_w}")
    print(f"b final: {b:.2f}")

    print(f"\nPredicciones vs Valores reales (entrenamiento):")
    for i in range(5): # Print for first 5 samples
        pred = hyp_theta(df_x_train[i], theta_w, b)
        print(f"-> Pred: {pred:.4f}, Real: {df_y_train[i]}")

if __name__ == "__main__":
    main()