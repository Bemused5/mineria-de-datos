import pandas as pd

df = pd.read_csv('/Volumes/ExtremeSSD/Universidad/DecimoSemestre/MineriaDatos/mineria-de-datos/ExamenFinal/items.csv')

df.eval('importe = precio * cantidad', inplace=True)

dfa = df.query('categoria == "A"')

promedio_por_categoria = df.groupby('categoria')['importe'].mean()

promedio_A = promedio_por_categoria.get('A', 0)
promedio_A_redondeado = round(promedio_A, 3)

print(f"Promedio de importe de la categoría A: {promedio_A_redondeado}")
