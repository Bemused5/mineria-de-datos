import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sensores.csv")

# Carga el CSV
df = pd.read_csv(csv_path)

# Selecciona solo columnas
df = df[['ts', 'temperatura', 'humedad']]

# Convierte ts a datetime
df['ts'] = pd.to_datetime(df['ts'], format='mixed', errors='coerce')

# Calcula mediana y media originales para la imputación
mediana_temp_orig = df['temperatura'].median()
media_hum_orig = df['humedad'].mean()

# Reemplaza valores faltantes en temperatura con la mediana y en humedad con la media
df['temperatura'] = df['temperatura'].fillna(mediana_temp_orig)
df['humedad'] = df['humedad'].fillna(media_hum_orig)

# Reporte
# (a) ¿Cuántos NaT quedaron en ts?
a_nat_en_ts = df['ts'].isna().sum()

# (b) Valor final de la mediana de temperatura tras imputación (redondea a 1 decimal)
b_mediana_temp_final = round(df['temperatura'].median(), 1)

# (c) Valor final de la media de humedad tras imputación (redondea a 3 decimales)
c_media_hum_final = round(df['humedad'].mean(), 3)

print("Resultados:")
print(f"(a) NaT en ts: {a_nat_en_ts}")
print(f"(b) Mediana final de temperatura: {b_mediana_temp_final}")
print(f"(c) Media final de humedad: {c_media_hum_final}")
print(f"Formato de entrega: {a_nat_en_ts}, {b_mediana_temp_final}, {c_media_hum_final}")
