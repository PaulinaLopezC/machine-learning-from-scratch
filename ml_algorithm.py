# ---------------------------------------------------------------------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
# Guadalupe Paulina López Cuevas
# A01701095
# 
# Linear Regression Model + Logaritmic Cost Function
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# Funcion para dividir el dataset en train y test y sacar las Xs y Ys
def divide_train_test(df, x_columnas):
    # Separar dataset en train y test
    # 80% para train
    train_size = int(0.8 * len(df))
    # 20% para test
    test_size = len(df) - train_size
    # Dividir el dataset en train y test
    df_train = df[:train_size]
    df_test = df[train_size:]
    # Escoger mi dataset para train y test
    x_train = df_train[x_columnas].values.astype(np.float64)
    y_train = df_train['total_UPDRS'].values.astype(np.float64)

    x_test = df_test[x_columnas].values.astype(np.float64)
    y_test = df_test['total_UPDRS'].values.astype(np.float64)
    return x_train, y_train, x_test, y_test

# Funcion para hacer transfornacion de variables dependientes e independientes
def trans_dep_indep(x_train, y_train, x_test, y_test):
  df_y_train = y_train.T
  df_y_test = y_test.T
  scaler = StandardScaler()
  df_x_train = scaler.fit_transform(x_train)
  df_x_test = scaler.transform(x_test)
  return df_x_train, df_y_train, df_x_test, df_y_test

def hyp(x, theta_w, b):
    # Y = b + x*theta_w
    # Multiplicacion entre matrices
    return b + (x@theta_w)

# Funcion para sacar mi Mean Square Error
def MSE(x, y, theta_w, b):
    cost = 0
    m = len(y)
    y_hyp = hyp(x, theta_w, b)
    cost = (y_hyp - y)**2
    mean_cost = np.mean(cost)
    return mean_cost

def update_gradients(x, theta_w, b, y, alfa):
    # Convertir a numpy para mayor eficiencia
    x = np.array(x)
    theta_w = np.array(theta_w)
    Y = np.array(y)
    
    m = len(x)  # número de ejemplos
    n = len(theta_w)  # número de features
    
    # Calcular predicciones para todas las muestras
    predictions = hyp(x,theta_w, b)  # Vectorizado: más eficiente
    # Calcular errores
    errors = predictions - y
    # Actualizar theta (gradiente para cada parámetro)
    grad = grad + np.dot(x.T, errors)
    theta_new = theta_w - (alfa / m) * grad
    # Actualizar bias
    b_new = b - (alfa / m) * np.sum(errors)
    return theta_new, b_new

def GD():
    pass

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


    # Separar nuestro dataset en train y test
    x_columnas = ['age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    x_train, y_train, x_test, y_test = divide_train_test(df_shuffle,x_columnas)
    # Transformacion para variables dependientes e independientes
    df_x_train, df_y_train, df_x_test, df_y_test = trans_dep_indep(x_train, y_train, x_test, y_test)
    print("VARIABLES DEPENDIENTES\n")
    print("Dimensiones Y train\n", df_y_train.shape)
    print("Dimensiones Y test\n", df_y_test.shape)
    print("VARIABLES INDEPENDIENTES\n")
    print("Scale X train\n",df_x_train)
    print("Scale X test\n", df_x_test)


    print("INICIALIZAR PARAMETROS")
    # Weights
    theta_w = np.zeros(len(x_columnas))
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
    epoch = 10
    print("Epochs = ", epoch)



if __name__ == "__main__":
    main()