
import numpy as np
import pandas as pd
from io import StringIO

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

# Cargar datos
df = pd.read_csv(StringIO(raw), sep=',')
df.columns = [c.strip() for c in df.columns]

# Variables
df['edad'] = pd.to_numeric(df['¿Cuántos años cumplidos tienes?'], errors='coerce')
map_sn = {'Si': True, 'No': False}
df['prog_python'] = df['¿Has realizado programas utilizando Python?'].map(map_sn)
df['usa_colab'] = df['¿Has utilizado Google Colab?'].map(map_sn)
df['cuenta_github'] = df['¿Tienes cuenta de GitHub?'].map(map_sn)

# A NumPy
edad = df['edad'].to_numpy(dtype=float)
prog_py = df['prog_python'].to_numpy(dtype=bool)
usa_colab = df['usa_colab'].to_numpy(dtype=bool)
cuenta_gh = df['cuenta_github'].to_numpy(dtype=bool)

# Descriptivos de edad (NumPy)
summary_age = {
    'n': int(edad.size),
    'min': float(np.min(edad)),
    'p25': float(np.percentile(edad, 25)),
    'mediana': float(np.median(edad)),
    'media': float(np.mean(edad)),
    'p75': float(np.percentile(edad, 75)),
    'max': float(np.max(edad)),
    'rango(ptp)': float(np.ptp(edad)),
    'var_muestral': float(np.var(edad, ddof=1)),
    'std_muestral': float(np.std(edad, ddof=1)),
    'cv': float(np.std(edad, ddof=1)/np.mean(edad)),
}

# Frecuencias (NumPy)
freq_python = {'Si': int(np.count_nonzero(prog_py)), 'No': int(edad.size - np.count_nonzero(prog_py))}
freq_colab  = {'Si': int(np.count_nonzero(usa_colab)), 'No': int(edad.size - np.count_nonzero(usa_colab))}
freq_gh     = {'Si': int(np.count_nonzero(cuenta_gh)), 'No': int(edad.size - np.count_nonzero(cuenta_gh))}

# Tablas cruzadas (NumPy)
def tabla2x2(a, b):
    t = np.zeros((2,2), dtype=int)
    t[1,1] = np.count_nonzero(a & b)
    t[1,0] = np.count_nonzero(a & ~b)
    t[0,1] = np.count_nonzero(~a & b)
    t[0,0] = np.count_nonzero(~a & ~b)
    return t

tab_py_colab = tabla2x2(prog_py, usa_colab)
tab_py_gh    = tabla2x2(prog_py, cuenta_gh)
tab_colab_gh = tabla2x2(usa_colab, cuenta_gh)

# Medias de edad por grupo (NumPy)
means_by_group = {
    'edad_promedio_Python': {'Si': float(np.mean(edad[prog_py])) if np.any(prog_py) else np.nan,
                             'No': float(np.mean(edad[~prog_py])) if np.any(~prog_py) else np.nan},
    'edad_promedio_Colab':  {'Si': float(np.mean(edad[usa_colab])) if np.any(usa_colab) else np.nan,
                             'No': float(np.mean(edad[~usa_colab])) if np.any(~usa_colab) else np.nan},
    'edad_promedio_GitHub': {'Si': float(np.mean(edad[cuenta_gh])) if np.any(cuenta_gh) else np.nan,
                             'No': float(np.mean(edad[~cuenta_gh])) if np.any(~cuenta_gh) else np.nan},
}

summary_age, freq_python, freq_colab, freq_gh, tab_py_colab, tab_py_gh, tab_colab_gh, means_by_group