import pandas as pd
import os

# Asegurar que se lea el CSV con la ruta correcta relativa a este archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "items.csv")

# 1. Cargar el CSV en un DataFrame df.
df = pd.read_csv(csv_path)

# 2. Con eval(), crea la columna importe = precio * cantidad.
df.eval('importe = precio * cantidad', inplace=True)

# 3. Con query(), filtra solo categoria == 'A' y guarda en dfa.
dfa = df.query("categoria == 'A'")

# 4. Con groupby(), calcula el promedio de importe por categoria en el DataFrame original df (no filtrado).
promedio_por_categoria = df.groupby('categoria')['importe'].mean()

promedio_A = round(promedio_por_categoria.get('A', 0), 3)

print("Promedios por categoría:\n", promedio_por_categoria)
print("\nEl promedio de importe de la categoría A (redondeado a 3 decimales) es:", promedio_A)
