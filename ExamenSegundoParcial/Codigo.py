import pandas as pd
import os

# Obtiene la ruta del directorio donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construye la ruta absoluta al archivo CSV
csv_path = os.path.join(script_dir, "ventas.csv")
 
df = pd.read_csv(csv_path, parse_dates=['fecha'])

df = df.set_index('fecha')

print(df.head(3))

print(df.info())
