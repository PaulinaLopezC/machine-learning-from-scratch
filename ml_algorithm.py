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
    df = pd.read_csv('predict_students_success_data.csv', sep=';')
    print(df)
    return df

def main():
    df = load_data('predict_students_success_data.csv')

if __name__ == "__main__":
    main()