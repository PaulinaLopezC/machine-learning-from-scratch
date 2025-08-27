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
import math

def load_data(file_path):
    # Cargar el data set desde el .csv para manipular los datos
    df = pd.read_csv(file_path, sep=',')
    print("LOADED DATASET\n", df)
    return df

def null_count(df):
    df_nulos = df.isnull().mean()*100
    print("NULL DATA\n", df_nulos)
    return df_nulos

def duplicate_exist(df):
  df_nonduplicate = df.drop_duplicates()
  print("NO MORE DUPLICATED\n", df_nonduplicate)
  return df_nonduplicate
    
def remove_subject_column(df, column):
    return df.drop(column, axis=1)

def outliers_drop(df, columns):
  for i in columns:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]
  return df

def clean_dataset(df):
    df_nulos = null_count(df)
    df.info()
    df.describe()
    df = duplicate_exist(df)
    df_clean = remove_subject_column(df, 'subject#')
    df_clean = remove_subject_column(df_clean, 'test_time')
    columns =  ['Jitter(%)', 'Jitter(Abs)', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    df_clean = outliers_drop(df_clean, columns)
    return df_clean


def main():
    df = load_data('parkinsons_updrs.data')
    df_clean = clean_dataset(df)
    df_x = df_clean[['age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
    df_y1 = df_clean[['total_UPDRS']]
    df_y2 = df_clean[['motor_UPDRS']]
    

if __name__ == "__main__":
    main()