import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
path_clientes = os.path.join(script_dir, "clientes.csv")
path_ordenes = os.path.join(script_dir, "ordenes.csv")

# 1. Lee ambos CSVs
dfc = pd.read_csv(path_clientes)
dfo = pd.read_csv(path_ordenes)

# 2. Left merge de dfc con dfo por cliente_id
df_merged = dfc.merge(dfo, on="cliente_id", how="left")

# (a) Número total de filas
total_filas = len(df_merged)

# (b) Monto total por cliente (suma de monto) y reemplaza NaN por 0
df_merged["monto"] = df_merged["monto"].fillna(0)
monto_por_cliente = df_merged.groupby("cliente_id")["monto"].sum()

print(f"Total de filas del left merge: {total_filas}")
print("\nMonto total por cliente:")
print(monto_por_cliente)

monto_cliente_1 = monto_por_cliente.get(1, 0)
print(f"\nMonto total para el cliente 1: {monto_cliente_1}")
