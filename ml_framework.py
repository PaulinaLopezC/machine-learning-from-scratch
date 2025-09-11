# ---------------------------------------------------------------------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje máquina con el uso de un framework y mejorarlo. (Portafolio Implementación)
# Guadalupe Paulina López Cuevas
# A01701095
# 
# Random Forest
# ---------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from scipy import stats

import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# LIMPIEZA DE DATOS
# ==================================================================================
# Funcion para cargar el data set desde un archivo para manipular los datos
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    print("LOADED DATASET\n", df)
    return df

# Funcion que nos sirve para remover todos los datos nulos encontrados en el dataset, de ser necesario
def remove_nulls(df):
    df_clean = df.dropna()
    return df_clean

# Funcion para revisar si existe algun valor duplicado y quitarlo del dataset
def remove_duplicate(df):
  df_nonduplicate = df.drop_duplicates()
  print("NO MORE DUPLICATED\n", df_nonduplicate)
  return df_nonduplicate

# Funcion para quitar alguna colunma en especifico
# Objetivo: remover datos del dataset que no sean considerados relevantes para nuestro entrenamiento
def remove_subject_column(df, column):
    return df.drop(column, axis=1)

# ==================================================================================
# DIVISION DEL DATASET EN TRAIN Y TEST
# ==================================================================================
# Funcion para dividir y deplegar shape del dataset de train y test
def divide_train_val_test(df, y_target, test_size=0.2, validation_size=0.2, random_state=42):
    x = df.drop(y_target, axis=1)
    y = df[y_target]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(test_size + validation_size), random_state=random_state)
    test_size_adjusted = test_size / (test_size + validation_size)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_size_adjusted, random_state=random_state)

    return x_train, y_train, x_val, y_val, x_test, y_test

# ==================================================================================
# ESCALAMIENTO DE VARIABLES INDEPENDIENTES Y REDIMENSION DE VARIABLES DEPENDIENTES
# ==================================================================================
# Funcion para hacer transformacion de variables dependientes e independientes
def trans_indep(x_train,x_val, x_test):
    scaler = StandardScaler()
    df_x_train = scaler.fit_transform(x_train)
    df_x_val = scaler.fit_transform(x_val)
    df_x_test = scaler.transform(x_test)
    return df_x_train, df_x_val, df_x_test

# ==================================================================================
# GRAFICAS PARA VISUALIZAR ERROR, PREDICCION VS VALOR REAL (TRAIN vs VAL)
# ==================================================================================
def plot_train_vs_val(sample_sizes, train_scores, val_scores, train_acc, val_acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sample_sizes, train_scores, label='Train')
    axes[0].plot(sample_sizes, val_scores, label='Val')
    axes[0].set_xlabel('Tamaño de Muestra')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Random Forest MSE (train vs val)')
    axes[0].legend()

    axes[1].plot(sample_sizes, train_acc, label='Train')
    axes[1].plot(sample_sizes, val_acc, label='Val')
    axes[1].set_xlabel('Tamaño de Muestra')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Random Forest Accuracy (train vs val)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_train_vs_val_vs_test(sample_sizes, train_scores, val_scores, test_scores, train_acc, val_acc, test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sample_sizes, train_scores, label='Train')
    axes[0].plot(sample_sizes, val_scores, label='Val')
    axes[0].plot(sample_sizes, test_scores, label='Test')
    axes[0].set_xlabel('Tamaño de Muestra')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Random Forest MSE (train vs val vs test)')
    axes[0].legend()

    axes[1].plot(sample_sizes, train_acc, label='Train')
    axes[1].plot(sample_sizes, val_acc, label='Val')
    axes[1].plot(sample_sizes, test_acc, label='Test')
    axes[1].set_xlabel('Tamaño de Muestra')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Random Forest Accuracy (train vs val vs test)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    

def main():
    df = load_data('parkinsons_updrs.data')

    # ==================================================================================
    # LIMPIEZA DE DATOS
    # ==================================================================================
    # Remover los datos nulos si es que hay alguno
    df = remove_nulls(df)
    # Remover los valores duplicados
    df_clean = remove_duplicate(df)
    # Quitar columna 'subject#'
    df_clean = remove_subject_column(df_clean, 'subject#')
    # Quitar columna 'test_time'
    df_clean = remove_subject_column(df_clean, 'test_time')
    # Quitar columna 'motor_UPDRS'
    df_clean = remove_subject_column(df_clean, 'motor_UPDRS')
    # Revolver el dataset para tener los datos desordenados
    df_shuffle = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    # ==================================================================================
    # DIVISION DEL DATASET A TRAIN Y TEST, ESCALAMIENTO Y REDIMENSION
    # ==================================================================================
    # Dividir el dataset en train y test
    y_target = 'total_UPDRS'
    x_train, y_train, x_val, y_val, x_test, y_test = divide_train_val_test(df_shuffle, y_target)
    # Transformacion para variables dependientes e independientes
    X_train, X_val, X_test = trans_indep(x_train, x_val, x_test)

    # ==================================================================================
    # ARREGLOS PARA CURVA DE APRENDIZAJE DEL MODELO
    # ==================================================================================
    train_error = []
    val_error = []
    train_acc = []
    val_acc = []
    sample_sizes = range(50, len(X_train), 50)

    # ==================================================================================
    # ENTRENAMIENTO DEL MODELO
    # ==================================================================================
    for size in sample_sizes:
        model_rf = RandomForestRegressor(n_estimators=400, max_leaf_nodes=10, n_jobs=-1, random_state=42)
        model_rf.fit(X_train[:size], y_train[:size])

        y_train_pred_subset = model_rf.predict(X_train[:size])
        y_val_pred = model_rf.predict(X_val)

        train_error.append(mean_squared_error(y_train[:size], y_train_pred_subset))
        val_error.append(mean_squared_error(y_val, y_val_pred))
        train_acc.append(r2_score(y_train[:size], y_train_pred_subset))
        val_acc.append(r2_score(y_val, y_val_pred))
        
    # ==================================================================================
    # GRAFICAS PARA VISUALIZAR ERROR y ACCURACY (TRAIN vs VAL) 
    # ==================================================================================
    plot_train_vs_val(sample_sizes, train_error, val_error, train_acc, val_acc)

    y_train_pred_full = model_rf.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred_full)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred_full)
    val_r2 = r2_score(y_val, y_val_pred)

    print("RANDOM FOREST MODEL V1")
    print("MSE Train", train_mse)
    print("MSE Val", val_mse)
    print("R² Train", train_r2)
    print("R² Val", val_r2)
    
    
    # ==================================================================================
    # ==================================================================================

    # ==================================================================================
    # ARREGLOS PARA CURVA DE APRENDIZAJE DEL MODELO MEJORADO
    # ==================================================================================
    train_error = []
    val_error = []
    test_error = []
    train_acc = []
    val_acc = []
    test_acc = []
    sample_sizes = range(50, len(X_train), 50)

    # ==================================================================================
    # ENTRENAMIENTO DEL MODELO MEJORADO
    # ==================================================================================
    for size in sample_sizes:
        model_rf_v2 = RandomForestRegressor(n_estimators=200, max_depth= 15, max_features= 0.8, min_samples_leaf= 2, min_samples_split= 5, random_state=42)
        model_rf_v2.fit(X_train[:size], y_train[:size])

        y_train_pred_subset = model_rf_v2.predict(X_train[:size])
        y_val_pred = model_rf_v2.predict(X_val)
        y_test_pred = model_rf_v2.predict(X_test)

        train_error.append(mean_squared_error(y_train[:size], y_train_pred_subset))
        val_error.append(mean_squared_error(y_val, y_val_pred))
        test_error.append(mean_squared_error(y_test, y_test_pred))
        train_acc.append(r2_score(y_train[:size], y_train_pred_subset))
        val_acc.append(r2_score(y_val, y_val_pred))
        test_acc.append(r2_score(y_test, y_test_pred))
    
    # ==================================================================================
    # GRAFICAS PARA VISUALIZAR ERROR y ACCURACY (TRAIN vs VAL vs TEST) 
    # ==================================================================================
    plot_train_vs_val_vs_test(sample_sizes, train_error, val_error, test_error, train_acc, val_acc, test_acc)
    
    y_train_pred_full = model_rf_v2.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred_full)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred_full)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("RANDOM FOREST MODEL V2")
    print(f"MSE Train:", train_mse)
    print(f"MSE Val:", val_mse)
    print(f"MSE Test:", test_mse)
    print(f"R² Train:", train_r2)
    print(f"R² Val:", val_r2)
    print(f"R² Test:", test_r2)

if __name__ == "__main__":
    main()