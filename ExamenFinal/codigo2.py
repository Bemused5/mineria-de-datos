import pandas as pd

df = pd.read_csv('/Volumes/ExtremeSSD/Universidad/DecimoSemestre/MineriaDatos/mineria-de-datos/ExamenFinal/sensores.csv', usecols=['ts', 'temperatura', 'humedad'])

df['ts'] = pd.to_datetime(df['ts'], format='mixed', errors='coerce')

mediana_temp = df['temperatura'].median()
media_humedad = df['humedad'].mean()

df['temperatura'] = df['temperatura'].fillna(mediana_temp)
df['humedad'] = df['humedad'].fillna(media_humedad)

nat_count = df['ts'].isna().sum()

final_median_temp = df['temperatura'].median()

final_mean_humedad = df['humedad'].mean()

print(f"{nat_count}, {final_median_temp:.1f}, {final_mean_humedad:.3f}")
