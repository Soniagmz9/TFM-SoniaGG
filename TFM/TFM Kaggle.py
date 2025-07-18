# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:20:02 2025

@author: sonia
"""

# TFM Kaggle

# ----------------------------------------------------------------------------

# PREPROCESADO Y PREPARACIÓN DE LOS DATOS

# Importacion de librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer

# Importacion del dataset limpio
df = pd.read_excel('df_limpio.xlsx')
print(df.head())

# Almacenamiento de las variables independientes en x y la variable dependiente en y
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Division en conjuntos de entrenamiento y test
x_entrenamiento,x_prueba,y_entrenamiento,y_prueba = train_test_split(x,y,test_size=0.25,random_state=0)

# Correlacion entre variables
matriz_correlaciones = x.corr()
print(matriz_correlaciones)

sns.heatmap(matriz_correlaciones,annot=False,cmap='coolwarm')
plt.show()

# Escalado de los variables
sc = StandardScaler()
x_entrenamiento = sc.fit_transform(x_entrenamiento)
x_prueba = sc.transform(x_prueba)

# -------------------------------------------------------------------------------

# Seleccion de las variables el modelo - RANDOM FOREST
clasificador = RandomForestClassifier(n_estimators=100,criterion = 'entropy',random_state = 0)
clasificador.fit(x_entrenamiento,y_entrenamiento)

y_prediccion = clasificador.predict(x_prueba)
significacion_variables = clasificador.feature_importances_
nombres_variables = x.columns.tolist()

# ------------------------------------------------------------------------------

# Representacion de las variables para saber cuales iran dentro del modelo final
plt.figure(figsize=(10,8))
plt.barh(nombres_variables,significacion_variables, color = 'green')
plt.axvline(x=0.05,color = 'red')
plt.xlabel('Significación')
plt.title('Significación de las variables - Random Forest')
plt.tight_layout()
plt.show()

# Bucle para eliminar variables con importancia mayor a 0.05
# while max(significacion_variables) >= 0.05 and len(x.columns) > 0:
    # Obtener el índice de la variable con mayor importancia
    # indice_max = significacion_variables.argmax()
    
    # Imprimir el nombre de la variable antes de eliminarla
    # print(f"Eliminando la variable: {x.columns[indice_max]}")
    
    # Eliminar la variable del DataFrame
    # x = x.drop(columns=[x.columns[indice_max]])
    
    # Verificar que el DataFrame no esté vacío antes de entrenar el modelo
    # if len(x.columns) == 0:
        # print("No quedan variables para entrenar el modelo.")
        # break
    
    # Entrenar el modelo nuevamente con las variables restantes
    # clasificador.fit(x, y)
    
    # Obtener la nueva importancia de las variables
    # significacion_variables = clasificador.feature_importances_

# Crear el gráfico de barras horizontales con las variables restantes
# if len(x.columns) > 0:
    # plt.figure(figsize=(10, 8))  # Ajustar el tamaño de la figura
    # plt.barh(x.columns, significacion_variables, color='green')
    # plt.axvline(x=0.05, color='red', linestyle='--')  # Agregar la línea vertical en el valor 0.05
    # plt.xlabel('Significación')
    # plt.title('Significación de las variables - Random Forest')
    # plt.tight_layout()  # Ajustar el espaciado para evitar que se junten
    # plt.show()
# else:
    # print("No hay variables con importancia menor a 0.05 para graficar.")

# --------------------------------------------------------------------------

# ENTRENAMIENTO DEL PREDICTOR

# RANDOM FOREST
# creacion y aplicacion del clasificador
regression = RandomForestClassifier(n_estimators=300,random_state=0)
regression.fit(x_entrenamiento,y_entrenamiento)

# prediccion con el conjunto de prueba
y_prediccion = regression.predict(x_prueba)

# metricas
print('MÉTRICAS RANDOM FOREST:')
cm = confusion_matrix(y_prueba, y_prediccion)
print("Matriz de Confusión:")
print(cm)
accuracy = accuracy_score(y_prueba, y_prediccion)
print(f'Accuracy: {round(accuracy,4)}')


# REGRESION LOGISTICA
# creacion del clasificador, con la semilla en 0 para que se pueda replicar
clasificador = LogisticRegression(random_state=0,multi_class='multinomial')
clasificador.fit(x_entrenamiento,y_entrenamiento)

# Prediccion con el conjunto de prueba
y_prediccion = clasificador.predict(x_prueba)


# Metricas
# Matriz de confusion
print('MÉTRICAS REGRESIÓN LOGÍSTICA:')
matriz_confusion = confusion_matrix(y_prueba,y_prediccion)
print(matriz_confusion)

# Accuracy
Accuracy = accuracy_score(y_prueba,y_prediccion)
print(f'Accuracy: {round(Accuracy,4)}')

# --------------------------------------------------------------------------

# RED NEURONAL - LSTM

# vectorizacion de x e y 
vectorizador = TfidfVectorizer()
x= vectorizador.fit_transform(x).toarray()

# Escalado de los datos
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)




























