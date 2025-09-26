import pandas as pd 
import numpy as np
import seaborn as sns
import os 
import re
import matplotlib
matplotlib.use("Agg")  # Backend pour génération sans GUI
matplotlib.rcParams.update({"font.size":12})
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Fitness.EEM import RE,ABS,RMSE,IE,RE_A,_preprocess_data




# Paramètres
detailed = pd.read_csv("Processing_Detailed.csv")
list_species = ["Y_H2O","Y_NH3","Y_OH","Y_HNO"]  # mettre tes espèces
notation=[r"$Y_{H_2O}$",r"$Y_{NH_3}$",r"$Y_{OH}$",r"$Y_{HNO}$"]
# list_species = ["Y_NH3"]
X = 50
T = 1000
Phi = 0.8
epsilon = 5e-4

df_subset = detailed.loc[
    (detailed["T_Init"] == T) & (detailed["Phi_Init"] == Phi),
    list_species+["common_grid","T_Init","Phi_Init"]
].iloc[X:]
time = df_subset["common_grid"].values

# Nouvelle grille temporelle
time_forward = time + epsilon
time_backward = time - epsilon

# Initialisation des DataFrames
df_forward = pd.DataFrame({"common_grid": time_forward})
df_backward = pd.DataFrame({"common_grid": time_backward})

dt = time[1] - time[0]  # résolution temporelle
shift_idx = int(round(epsilon / dt))

df_forward = df_subset.copy()
df_backward = df_subset.copy()

# Décaler les valeurs par indices
for sp in list_species:
    df_forward[sp].iloc[:-shift_idx] = df_subset[sp].iloc[shift_idx:].values
    df_forward[sp].iloc[-shift_idx:] = df_subset[sp].iloc[-1]  # ou np.nan

    df_backward[sp].iloc[shift_idx:] = df_subset[sp].iloc[:-shift_idx].values
    df_backward[sp].iloc[:shift_idx] = df_subset[sp].iloc[0]  # ou np.nan
# # Interpolation pour chaque espèce
# for sp in list_species:
#     values = df_subset[sp].values  # valeurs correspondant à la grille time
#     f_interp = interp1d(time, values, kind="cubic", fill_value="extrapolate")
#     df_forward[sp] = f_interp(time_forward)
#     df_backward[sp] = f_interp(time_backward)

# Ajouter les colonnes de condition
df_forward["T_Init"] = T
df_forward["Phi_Init"] = Phi
df_forward["common_grid"] =time
df_backward["T_Init"] = T
df_backward["Phi_Init"] = Phi
df_backward["common_grid"] =time

# listes des métriques et des directions
metrics = ["RE", "ABS", "RMSE", "RE_A"]
directions = {"fwd": df_forward, "bwd": df_backward}

# 1️⃣ Sans log ni normalisation
all_errors = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_errors[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True)

# 2️⃣ Avec log
all_errors_log = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_errors_log[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True, do_log=True)

# 3️⃣ Normalisation Standard
all_error_norm_S = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_error_norm_S[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True, norm_type="Standard")

# 4️⃣ Log + Normalisation Standard
all_error_log_norm_S = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_error_log_norm_S[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True, do_log=True, norm_type="Standard")

# 3️⃣ Normalisation Standard
all_error_norm_M = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_error_norm_M[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True, norm_type="minmax")

# 4️⃣ Log + Normalisation Standard
all_error_log_norm_M = {}
for m in metrics:
    for dname, ddata in directions.items():
        key = f"{m}_{dname}"
        all_error_log_norm_M[key] = globals()[f"{m}"](df_subset, ddata, list_species, flag_output=True, do_log=True, norm_type="minmax")


# Paramètres
error_types = ["RE","ABS","RMSE","RE_A"]
species_cols = list_species[:4]  # 4 espèces pour le 2x2
n = 6                     # nombre de types
width = 0.15              # largeur de chaque barre
offsets = np.linspace(- (n-1)/2*width, (n-1)/2*width, n)

# Labels et couleurs
labels = ["Normal", "Log", "Std", "Log+Std","Mm","Log+Mm"]
colors = ["#F3DFA2", "#7EBDC2", "#9D8179", "#BB4430", "#6B9C82", "#C58C85"]

# Liste des dictionnaires pour les 4 combinaisons
dicts_ratio = [all_errors, all_errors_log, all_error_norm_S, all_error_log_norm_S,all_error_norm_M,all_error_log_norm_M]

# Création figure 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, sp in enumerate(species_cols):
    ax = axes[i]
    x = np.arange(len(error_types))

    for j in range(len(dicts_ratio)):
        vals = []
        for err in error_types:
            val_fwd = dicts_ratio[j][f"{err}_fwd"].set_index(["T_Init","Phi_Init"])[sp].values[0]
            val_bwd = dicts_ratio[j][f"{err}_bwd"].set_index(["T_Init","Phi_Init"])[sp].values[0]
            ratio = val_bwd / val_fwd
            vals.append(ratio)
        ax.bar(x + offsets[j], vals, width, color=colors[j], alpha=0.85,
               label=labels[j] if i==0 else "")

    # ax.set_yscale("log")
    ax.set_title(sp, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim([0, 2])
    ax.set_ylabel(r"$\epsilon(Bwd)/\epsilon(Fwd)$",fontsize=12)
    # Afficher les labels x sur tous les axes
    ax.set_xticks(np.arange(len(error_types)))
    ax.set_xticklabels(error_types, rotation=45, fontsize=15)

# Légende globale au-dessus
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=3, fontsize=12, frameon=False)

plt.tight_layout(rect=[0,0,1,0.93])  # laisser de l'espace pour la légende
plt.savefig("FB_Ratio.png")



error_types = [ "ABS", "RMSE", "RE_A"]
transfo_labels = ["Normal", "Log", "Std", "Log+Std","Mm","Log+Mm"]
species_cols = list_species
dicts_ratio = [all_errors, all_errors_log, all_error_norm_S, all_error_log_norm_S,all_error_norm_M,all_error_log_norm_M]


def build_matrix(direction):
    """Construit un dict {error_type: DataFrame} pour fwd ou bwd"""
    matrices = {}
    for err in error_types:
        matrix = np.zeros((len(species_cols), len(transfo_labels)))
        for i, sp in enumerate(species_cols):
            for j, d in enumerate(dicts_ratio):  # transformations
                val = d[f"{err}_{direction}"].set_index(["T_Init","Phi_Init"])[sp].values[0]
                matrix[i, j] = val
        df = pd.DataFrame(matrix, index=notation, columns=transfo_labels)
        matrices[err] = df
    return matrices

fwd_matrices = build_matrix("fwd")
bwd_matrices = build_matrix("bwd")
def plot_four_heatmaps_1x4(matrices, title, filename):
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(16,6))
    
    # Préparer un min et max communs pour colorbar
    all_vals = np.concatenate([matrices[err].values.flatten() for err in error_types])
    norm = mcolors.LogNorm(vmin=1e-2, vmax=1e3)

    # Fonction pour formater les annotations
    def format_value(x):
        if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.1e}"
        else:
            return f"{x:.1e}"

    for k, err in enumerate(error_types):
        df = matrices[err]
        ax = axes[k]
        sns.heatmap(df, ax=ax, cmap="viridis", norm=norm,
                    cbar=False, annot=True, fmt="", annot_kws={"size":10,"rotation":45,"fontweight":"bold"})

        # Réécrire le texte annoté avec le format dynamique
        for text, val in zip(ax.texts, df.to_numpy().flatten()):
            text.set_text(format_value(val))

        ax.set_title(err, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis="y",labelsize=17)

    # Colorbar commune
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\epsilon$",size=17)

    # fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.13, hspace=0.4)
    plt.savefig(filename, dpi=300)
    plt.close()

# Création des PNG
plot_four_heatmaps_1x4(fwd_matrices, "Forward Errors", "heatmap_forward.png")
plot_four_heatmaps_1x4(bwd_matrices, "Backward Errors", "heatmap_backward.png")

