import pandas as pd 
import matplotlib
matplotlib.use("Agg")  # Backend pour génération sans GUI
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Chargement des données
data_REF = pd.read_csv("./CALCUL/ALL_SPECIES/Processing_Detailed.csv")
data_RED = pd.read_csv("./CALCUL/ALL_SPECIES/Processing_Reduced.csv")
nmech = 500 
data_AEDML = pd.read_csv(f"./CALCUL/ALL_SPECIES/Processing_AEDML_mech{nmech}.csv")
data_ORCH  = pd.read_csv(f"./CALCUL/ALL_SPECIES/Processing_ORCH_mech{nmech}.csv")
data_BMEAN = pd.read_csv(f"./CALCUL/ALL_SPECIES/Processing_BMEAN_mech{nmech}.csv")
data_BMAX  = pd.read_csv(f"./CALCUL/ALL_SPECIES/Processing_BMAX_mech{nmech}.csv")

# Espèces chimiques à tracer
species = ["Y_NH3","Y_H2","Y_O2","Y_H2O","Y_O","Y_H","Y_OH","Y_NO","Y_N2O"]

# Liste des ER et des xlim associés
ER_list = [0.5, 1.5, 6, 13]
xlims = {
    0.5: [0, 0.004],
    1.5: [0, 0.005],
    6: [0, 0.02]
}

# Styles lisibles
plot_styles = {
    "REF":   {"label": "Detailed",          "color": "black",     "linestyle": "-",  "marker": ""},
    "RED":   {"label": "Reduced",           "color": "#e41a1c",   "linestyle": "--", "marker": ""},
    "AEDML": {"label": f"AEDML_mech{nmech}", "color": "#4daf4a",   "linestyle": "-.", "marker": ""},
    "ORCH":  {"label": f"ORCH_mech{nmech}",  "color": "#a65628",   "linestyle": ":",  "marker": ""},
    "BMEAN": {"label": f"BMEAN_mech{nmech}", "color": "#984ea3",   "linestyle": "-",  "marker": ""},
    "BMAX":  {"label": f"BMAX_mech{nmech}",  "color": "#ff7f00",   "linestyle": "--", "marker": ""},
}

for ER in ER_list:
    loc_data = {
        "REF": data_REF[data_REF["Phi_Init"] == ER],
        "RED": data_RED[data_RED["Phi_Init"] == ER],
        "AEDML": data_AEDML[data_AEDML["Phi_Init"] == ER],
        "ORCH": data_ORCH[data_ORCH["Phi_Init"] == ER],
        "BMEAN": data_BMEAN[data_BMEAN["Phi_Init"] == ER],
        "BMAX": data_BMAX[data_BMAX["Phi_Init"] == ER]
    }

    n_species = len(species)
    n_cols = 3
    n_rows = (n_species + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False, constrained_layout=True)

    for i, s in enumerate(species):
        row, col = divmod(i, n_cols)
        ax = axs[row][col]
        
        for key, df in loc_data.items():
            if not df.empty:
                style = plot_styles[key]
                ax.plot(df["common_grid"], df[s],
                        label=style["label"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=4,
                        linewidth=2)

        ax.set_xlabel("Common time grid")
        ax.set_ylabel(s)
        ax.set_title(s, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

        # Application du xlim si défini
        if ER in xlims:
            ax.set_xlim(xlims[ER])
            
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Suppression des sous-graphiques inutilisés
    for i in range(n_species, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axs[row][col])

    # fig.suptitle(f"Species Profiles at ER = {ER}", fontsize=16)
    plt.tight_layout()#rect=[0, 0, 1, 0.95]
    plt.savefig(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Species_Processing_ER{ER}.png", dpi=300)
    plt.close()
