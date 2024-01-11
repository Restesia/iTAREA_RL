import  pandas as pd
from datetime import datetime, date, time, timedelta

df = pd.read_excel("/home/dosgran/Documentos/dcaosd/Pruebas/RL/utils/data_10.110.15.xlsx")

# Convertir la columna de fechas a objetos datetime
df['start time'] = pd.to_datetime(df['start time'], format='%Y-%m-%d %H:%M:%S')

# Ordenar el DataFrame por la columna de fechas
df = df.sort_values(by='start time')

# Guardar el DataFrame ordenado de nuevo en el archivo CSV
df.to_csv('dataset_shangai.csv', index=False)