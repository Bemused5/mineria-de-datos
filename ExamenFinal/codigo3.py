import pandas as pd

df = pd.read_csv("ventas.csv", parse_dates=['fecha'])

df = df.set_index('fecha')

print(df.head(3))

print(df.info())
