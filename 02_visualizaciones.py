# --- Configuración e importaciones ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Tamaño y estilo de gráficos
plt.rcParams["figure.dpi"] = 160
plt.style.use('seaborn-v0_8-whitegrid')

# --- Datos crudos ---
raw = """Marca temporal,Número de matrícula,¿Cuántos años cumplidos tienes?,¿Has realizado programas utilizando Python?,¿Has utilizado Google Colab?,¿Tienes cuenta de GitHub?
19/01/2026 18:32:18,178920,22,Si,No,Si
19/01/2026 18:32:23,182712,21,Si,No,Si
19/01/2026 18:32:25,180370,26,Si,Si,Si
19/01/2026 18:32:27,182570,21,Si,Si,Si
19/01/2026 18:32:28,177622,24,Si,No,Si
19/01/2026 18:32:29,175166,22,Si,Si,Si
19/01/2026 18:32:30,177573,22,Si,Si,Si
19/01/2026 18:32:35,178430,21,Si,Si,Si
19/01/2026 18:32:38,179169,23,Si,No,Si
19/01/2026 18:32:41,177263,22,Si,Si,Si
19/01/2026 18:32:48,179884,23,Si,No,Si
19/01/2026 18:32:48,183016,22,Si,No,Si
19/01/2026 18:32:50,179033,23,Si,Si,No
19/01/2026 18:32:54,178218,22,Si,No,Si
19/01/2026 18:32:55,178446,22,Si,Si,Si
19/01/2026 18:33:00,178166,22,Si,Si,No
19/01/2026 18:33:02,177406,22,Si,No,Si
19/01/2026 18:33:05,177192,22,Si,Si,Si
19/01/2026 18:33:08,182451,21,Si,No,Si
19/01/2026 18:33:08,171513,24,No,No,Si
19/01/2026 18:33:09,181619,21,Si,No,Si
19/01/2026 18:33:12,182298,23,Si,No,Si
19/01/2026 18:33:14,181419,24,No,No,Si
19/01/2026 18:33:32,177291,21,Si,Si,Si
19/01/2026 18:33:42,182085,22,Si,No,Si
19/01/2026 18:33:53,176453,22,Si,No,Si
19/01/2026 18:34:04,176263,24,Si,No,Si
19/01/2026 18:34:06,179997,21,Si,No,Si
19/01/2026 18:36:21,175842,23,Si,No,No
19/01/2026 20:19:21,179913,20,Si,No,Si
19/01/2026 20:19:48,177301,22,Si,Si,No
19/01/2026 20:20:06,183060,21,Si,No,Si
19/01/2026 20:20:08,177935,22,Si,Si,Si
19/01/2026 20:20:19,177700,20,Si,Si,Si
19/01/2026 20:20:26,178318,24,Si,Si,Si
19/01/2026 20:20:27,174653,24,Si,No,Si
19/01/2026 20:20:29,177139,21,Si,No,Si
19/01/2026 20:20:31,173479,25,Si,No,No
19/01/2026 20:20:39,175588,23,No,No,Si
19/01/2026 20:20:45,178774,23,Si,Si,Si
19/01/2026 20:20:45,175031,23,Si,No,No
19/01/2026 20:20:49,178396,21,Si,Si,Si
19/01/2026 20:20:49,172068,23,No,No,No
19/01/2026 20:20:50,174197,23,No,No,Si
19/01/2026 20:20:55,178584,22,Si,No,Si
19/01/2026 20:20:55,175329,24,Si,No,Si
19/01/2026 20:20:56,177685,22,Si,Si,Si
19/01/2026 20:20:57,178678,22,No,No,Si
19/01/2026 20:21:04,177143,22,Si,No,Si
19/01/2026 20:21:07,182318,26,Si,No,Si
19/01/2026 20:21:27,182377,20,Si,No,Si
19/01/2026 20:21:34,179419,22,Si,No,No
19/01/2026 20:21:35,181662,21,Si,No,Si
19/01/2026 20:21:38,181760,21,Si,No,Si
19/01/2026 20:22:06,177888,22,Si,No,Si
19/01/2026 20:22:15,176535,25,Si,No,Si
19/01/2026 20:22:29,179862,21,Si,No,Si
19/01/2026 20:22:32,178378,22,Si,No,Si
19/01/2026 20:22:50,177451,22,Si,No,Si
19/01/2026 20:23:15,179804,21,Si,No,Si
"""

# --- Carga y limpieza ---
df = pd.read_csv(StringIO(raw), sep=',')
df.columns = [c.strip() for c in df.columns]

# --- Variables derivadas ---
df['edad'] = pd.to_numeric(df['¿Cuántos años cumplidos tienes?'], errors='coerce')
map_sn = {'Si': True, 'No': False}
df['prog_python'] = df['¿Has realizado programas utilizando Python?'].map(map_sn)
df['usa_colab'] = df['¿Has utilizado Google Colab?'].map(map_sn)
df['cuenta_github'] = df['¿Tienes cuenta de GitHub?'].map(map_sn)

# --- Paleta de colores ---
accent  = '#2563eb'  # azul
accent2 = '#16a34a'  # verde
accent3 = '#f59e0b'  # ámbar
neutral = '#6b7280'  # gris

# =========================
# 1) Histograma de edades
# =========================
fig, ax = plt.subplots(figsize=(8,5))
bins = range(int(df['edad'].min())-1, int(df['edad'].max())+2)
ax.hist(df['edad'], bins=bins, color=accent, edgecolor='white', alpha=0.85)
mean_age = df['edad'].mean()
med_age = df['edad'].median()
ax.axvline(mean_age, color=accent2, linestyle='--', linewidth=2, label=f"Media: {mean_age:.2f}")
ax.axvline(med_age, color=accent3, linestyle=':', linewidth=2.5, label=f"Mediana: {med_age:.2f}")
ax.set_title('Distribución de edades')
ax.set_xlabel('Edad (años)')
ax.set_ylabel('Frecuencia')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig('01_hist_edades.png', dpi=180)

# =======================
# 2) Boxplot de edades
# =======================
fig, ax = plt.subplots(figsize=(6,4))
ax.boxplot(df['edad'], vert=False, patch_artist=True,
           boxprops=dict(facecolor=accent, color=accent),
           medianprops=dict(color='white', linewidth=2),
           whiskerprops=dict(color=accent), capprops=dict(color=accent))
ax.set_title('Caja y bigotes: edad')
ax.set_xlabel('Edad (años)')
plt.tight_layout()
plt.show()
fig.savefig('02_box_edades.png', dpi=180)

# ====================================================
# 3) Barras: frecuencias de Sí/No para Python/Colab/GitHub
# ====================================================
fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
for ax, col, title, color in zip(
    axes,
    ['prog_python','usa_colab','cuenta_github'],
    ['Programó en Python','Usó Google Colab','Cuenta de GitHub'],
    [accent, accent2, accent3]
):
    counts = df[col].value_counts().reindex([True, False]).fillna(0).astype(int)
    labels = ['Sí','No']
    ax.bar(labels, counts, color=[color, neutral], edgecolor='white')
    ax.set_title(title)
    ax.set_ylim(0, max(counts.max(), 1)+2)
    for i, v in enumerate(counts):
        ax.text(i, v+0.1, str(v), ha='center', va='bottom', fontsize=9)
axes[0].set_ylabel('Número de estudiantes')
plt.tight_layout()
plt.show()
fig.savefig('03_barras_frecuencias.png', dpi=180)

# =======================================
# 4) Mapas de calor 2x2 para combinaciones
# =======================================
pairs = [
    ('prog_python','usa_colab','Python vs Colab'),
    ('prog_python','cuenta_github','Python vs GitHub'),
    ('usa_colab','cuenta_github','Colab vs GitHub')
]

for idx, (a,b,titulo) in enumerate(pairs, start=1):
    a_bool = df[a].to_numpy(dtype=bool)
    b_bool = df[b].to_numpy(dtype=bool)
    t = np.zeros((2,2), dtype=int)
    t[1,1] = np.count_nonzero(a_bool & b_bool)
    t[1,0] = np.count_nonzero(a_bool & ~b_bool)
    t[0,1] = np.count_nonzero(~a_bool & b_bool)
    t[0,0] = np.count_nonzero(~a_bool & ~b_bool)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(t, cmap='Blues')
    ax.set_title(f'Tabla 2×2: {titulo}')
    ax.set_xticks([0,1]); ax.set_xticklabels(['No','Sí'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['No','Sí'])
    ax.set_xlabel(b); ax.set_ylabel(a)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, t[i,j], ha='center', va='center',
                    color='black', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Conteo', rotation=270, labelpad=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'04_heatmap_{idx}.png', dpi=180)

# ======================================================
# 5) Barras: edad promedio por grupo (Sí/No) por variable
# ======================================================
fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
for ax, col, title, color in zip(
    axes,
    ['prog_python','usa_colab','cuenta_github'],
    ['Edad promedio por Python','Edad promedio por Colab','Edad promedio por GitHub'],
    [accent, accent2, accent3]
):
    means = df.groupby(col)['edad'].mean().reindex([True, False])
    labels = ['Sí','No']
    bars = ax.bar(labels, means, color=[color, neutral], edgecolor='white')
    for i, v in enumerate(means):
        ax.text(i, v+0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel('Grupo')
axes[0].set_ylabel('Edad promedio (años)')
plt.tight_layout()
plt.show()
fig.savefig('05_barras_edad_promedio.png', dpi=180)

# ===================================
# 6) Estadísticos de edad (impresión)
# ===================================
summary_age = {
    'n': int(df['edad'].size),
    'min': float(df['edad'].min()),
    'p25': float(np.percentile(df['edad'], 25)),
    'mediana': float(df['edad'].median()),
    'media': float(df['edad'].mean()),
    'p75': float(np.percentile(df['edad'], 75)),
    'max': float(df['edad'].max()),
    'rango(ptp)': float(np.ptp(df['edad'])),
    'var_muestral': float(np.var(df['edad'], ddof=1)),
    'std_muestral': float(np.std(df['edad'], ddof=1)),
    'cv': float(np.std(df['edad'], ddof=1)/np.mean(df['edad'])),
}
summary_age