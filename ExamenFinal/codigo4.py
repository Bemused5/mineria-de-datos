import pandas as pd

dfc = pd.read_csv('clientes.csv')
dfo = pd.read_csv('ordenes.csv')

df_merged = pd.merge(dfc, dfo, on='cliente_id', how='left')

total_filas = len(df_merged)

monto_total_por_cliente = df_merged.groupby('cliente_id')['monto'].sum().fillna(0)

monto_cliente_1 = monto_total_por_cliente.get(1, 0)

print(f"Número total de filas: {total_filas}")
print(f"Monto total para cliente_id=1: {monto_cliente_1}")
