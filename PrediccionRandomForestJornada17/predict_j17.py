import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data we found
df = pd.read_csv('/Volumes/ExtremeSSD/Universidad/DecimoSemestre/MineriaDatos/mineria-de-datos/PrediccionRandomForest/dataset_liga_mx_jornadas_13_14_15_base.csv')

# Features and target
features = ['diff_puntos', 'diff_racha', 'diff_posicion']
X = df[features]
y = df['resultado']

# Train the model
rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
rf.fit(X, y)

# Current stats after J16
stats = {
    'Chivas': {'pts': 35, 'racha': 10, 'pos': 1},
    'Pumas': {'pts': 33, 'racha': 11, 'pos': 2},
    'Pachuca': {'pts': 31, 'racha': 7, 'pos': 3},
    'Cruz Azul': {'pts': 30, 'racha': 7, 'pos': 4},
    'Toluca': {'pts': 27, 'racha': 4, 'pos': 5},
    'América': {'pts': 25, 'racha': 11, 'pos': 6},
    'Atlas': {'pts': 23, 'racha': 10, 'pos': 7},
    'Tigres': {'pts': 22, 'racha': 6, 'pos': 8},
    'León': {'pts': 22, 'racha': 7, 'pos': 9},
    'Tijuana': {'pts': 22, 'racha': 7, 'pos': 10},
    'Necaxa': {'pts': 18, 'racha': 7, 'pos': 11},
    'Monterrey': {'pts': 18, 'racha': 3, 'pos': 12},
    'Atlético San Luis': {'pts': 18, 'racha': 8, 'pos': 13},
    'Querétaro': {'pts': 17, 'racha': 6, 'pos': 14},
    'FC Juárez': {'pts': 16, 'racha': 1, 'pos': 15},
    'Mazatlán': {'pts': 15, 'racha': 5, 'pos': 16},
    'Puebla': {'pts': 13, 'racha': 0, 'pos': 17},
    'Santos': {'pts': 9, 'racha': 0, 'pos': 18}
}

matches_j17 = [
    ('Puebla', 'Querétaro'),
    ('Pachuca', 'Pumas'),
    ('Tigres', 'Mazatlán'),
    ('Toluca', 'León'),
    ('Chivas', 'Tijuana'),
    ('América', 'Atlas'),
    ('FC Juárez', 'Atlético San Luis'),
    ('Santos', 'Monterrey'),
    ('Cruz Azul', 'Necaxa')
]

print("Predictions for Jornada 17:")
for local, visita in matches_j17:
    diff_puntos = stats[local]['pts'] - stats[visita]['pts']
    diff_racha = stats[local]['racha'] - stats[visita]['racha']
    diff_posicion = stats[visita]['pos'] - stats[local]['pos']
    
    X_pred = pd.DataFrame([[diff_puntos, diff_racha, diff_posicion]], columns=features)
    pred = rf.predict(X_pred)[0]
    
    res_str = ""
    if pred == 0: res_str = "Gana visitante"
    elif pred == 1: res_str = "Empatan"
    else: res_str = "Gana local"
    
    print(f"{local} vs {visita}: {res_str} (Pred: {pred}, Pts:{diff_puntos}, Racha:{diff_racha}, Pos:{diff_posicion})")
