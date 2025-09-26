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
from matplotlib.table import Table
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Fitness.EEM import RE,ABS,RMSE,RE_A,_preprocess_data

def make_label(err_name, do_log=False, norm_type=None):
    """
    err_name : 'ABS', 'RMSE', 'RE_A'
    do_log   : True/False
    norm_type: None, 'Standard', 'minmax'
    """
    # map pour norm
    norm_map = {"standard": "std", "minmax": "mm"}
    norm_suffix = norm_map.get(norm_type, None)

    label = err_name  # base

    if do_log and norm_suffix:      # log + norm
        label = fr"$\widehat{{{err_name}}}^{{{norm_suffix}}}$"
    elif do_log and not norm_suffix: # log seul
        label = fr"$\widehat{{{err_name}}}$"
    elif not do_log and norm_suffix: # norm seul
        label = fr"${err_name}^{{{norm_suffix}}}$"
    else:                           # rien
        label = err_name

    return label



# --- Chargement des fichiers ---
REF = pd.read_csv("Processing_Detailed.csv")
RED_A = pd.read_csv("Processing_Reduced.csv")
RED_B = pd.read_csv("Processing_OptimB.csv")




input_fitness = ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO']


transformations = {
    "none":      {"do_log": None, "norm_type": None},
    "log":       {"do_log": True, "norm_type": None},
    "std":       {"do_log": None, "norm_type": "standard"},
    "mm":        {"do_log": None, "norm_type": "minmax"},
    "log_std":   {"do_log": True,  "norm_type": "standard"},
    "log_mm":    {"do_log": True,  "norm_type": "minmax"},
}
F_obj = {"RE_A": RE_A, "ABS": ABS, "RMSE": RMSE}
i = 0 

for t_name, params in transformations.items():
    total = []
    ratio_dict = {}
    labels_y = []  # pour stocker les noms des erreurs
    for fname, F in F_obj.items():

        Err_A = F(REF.copy(), RED_A.copy(), input_fitness,
                  flag_output=True,
                  do_log=params["do_log"],
                  norm_type=params["norm_type"])
        Err_A = Err_A[input_fitness].sum()
 
        
    
        Err_B = F(REF.copy(), RED_B.copy(), input_fitness,
                  flag_output=True,
                  do_log=params["do_log"],
                  norm_type=params["norm_type"])
        Err_B = Err_B[input_fitness].sum()
        
        ratio = Err_B[input_fitness].values / Err_A[input_fitness].values
    
        
        total.append(Err_B[input_fitness].sum()/Err_A[input_fitness].sum())
        ratio_dict[fname] = ratio
        labels_y.append(make_label(fname,
                                   do_log=params["do_log"],
                                   norm_type=params["norm_type"]))

    ratio_df = pd.DataFrame(ratio_dict, index=input_fitness).T  # lignes=erreur
    ratio_df["Total"] = total
    ratio_df.index = labels_y  # remplace l’index par tes labels

    plt.figure(figsize=(len(input_fitness)*0.5 + 1.5, 3))
    ax = sns.heatmap(ratio_df, annot=True, fmt=".1f", cmap="coolwarm",
                     annot_kws={"size": 9}, center=1, vmin=0, vmax=2)
    plt.xticks(rotation=45)
    cbar = ax.collections[0].colorbar  # la colorbar créée par seaborn
    cbar.set_label(r"$\epsilon(B)/\epsilon(A)$")  # ton nom ici
    
    # Ajouter une ligne verticale pour séparer la dernière espèce de Total
    n_species = len(input_fitness)
    ax.axvline(n_species, color='white', linewidth=2)  # colonne Total = n_species (0-indexé)
    
    plt.tight_layout()
    plt.savefig(f"heatmap/{i}_heatmap_ratio_{t_name}.png", dpi=300)
    i += 1
    plt.close()


###############################################

REF = pd.read_csv("Processing_Detailed.csv")
RED_A = pd.read_csv("Processing_Reduced.csv")
RED_B = pd.read_csv("Processing_OptimB.csv")




input_fitness = ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO']


transformations = {
    # "Normal":    {"do_log": None,"norm_type" : None},
    "log":       {"do_log": True, "norm_type": None},
    "std":       {"do_log": None, "norm_type": "standard"},
    "mm":        {"do_log": None, "norm_type": "minmax"},
    "log_std":   {"do_log": True,  "norm_type": "standard"},
    "log_mm":    {"do_log": True,  "norm_type": "minmax"},
}

T_values = REF["T_Init"].unique() 
ER_values = REF["Phi_Init"].unique() 
species = "Y_NO"
os.makedirs(f"heatmap/Profile/{species}",exist_ok=True)

results = {}
for t_name, params in transformations.items():
    do_log = params["do_log"]
    norm_type = params["norm_type"]

    A_abs = ABS(REF.copy(),RED_A.copy(),input_fitness,True,do_log,norm_type)
    B_abs = ABS(REF.copy(),RED_B.copy(),input_fitness,True,do_log,norm_type)
    A_rmse= RMSE(REF.copy(),RED_A.copy(),input_fitness,True,do_log,norm_type)
    B_rmse  = RMSE(REF.copy(),RED_B.copy(),input_fitness,True,do_log,norm_type)
    A_rea= RE_A(REF.copy(),RED_A.copy(),input_fitness,True,do_log,norm_type)
    B_rea= RE_A(REF.copy(),RED_B.copy(),input_fitness,True,do_log,norm_type)

    
    REF_change , RED_A_change = _preprocess_data(REF.copy(),RED_A.copy(),input_fitness,do_log,norm_type)
    REF_change,RED_B_change = _preprocess_data(REF.copy(),RED_B.copy(),input_fitness,do_log,norm_type)


    results[t_name] = {
        'params': params,
        'A_abs': A_abs,
        'B_abs': B_abs,
        'A_rmse': A_rmse,
        'B_rmse': B_rmse,
        'A_rea': A_rea,
        'B_rea': B_rea,
        'REF': REF_change,
        'RED_A': RED_A_change,
        'RED_B': RED_B_change
    }

# maintenant 1 image par (T,ER) avec 6 sous-plots
for T in T_values:
    for ER in ER_values:

        n_transfo = len(transformations)
        fig, axes = plt.subplots(n_transfo, 2, figsize=(12, 3*n_transfo),
                                 gridspec_kw={'width_ratios':[3,1]})
        if n_transfo == 1:  # gestion cas 1
            axes = [axes]

        for idx, (t_name, store) in enumerate(results.items()):
            do_log = store['params']["do_log"]
            norm_type = store['params']["norm_type"]

            loc_ref = store['REF'][(store['REF']["Phi_Init"]==ER)&(store['REF']["T_Init"]==T)]
            loc_redA = store['RED_A'][(store['RED_A']["Phi_Init"]==ER)&(store['RED_A']["T_Init"]==T)]
            loc_redB = store['RED_B'][(store['RED_B']["Phi_Init"]==ER)&(store['RED_B']["T_Init"]==T)]

            loc_A_abs = store['A_abs'][(store['A_abs']["Phi_Init"]==ER)&(store['A_abs']["T_Init"]==T)][species].values[0]
            loc_B_abs = store['B_abs'][(store['B_abs']["Phi_Init"]==ER)&(store['B_abs']["T_Init"]==T)][species].values[0]
            loc_A_rmse = store['A_rmse'][(store['A_rmse']["Phi_Init"]==ER)&(store['A_rmse']["T_Init"]==T)][species].values[0]
            loc_B_rmse = store['B_rmse'][(store['B_rmse']["Phi_Init"]==ER)&(store['B_rmse']["T_Init"]==T)][species].values[0]
            loc_A_rea = store['A_rea'][(store['A_rea']["Phi_Init"]==ER)&(store['A_rea']["T_Init"]==T)][species].values[0]
            loc_B_rea = store['B_rea'][(store['B_rea']["Phi_Init"]==ER)&(store['B_rea']["T_Init"]==T)][species].values[0]

            row_labels = [make_label("RE_A",do_log,norm_type),
                          make_label("ABS",do_log,norm_type),
                          make_label("RMSE",do_log,norm_type)]

            # profil
            ax1 = axes[idx][0] if n_transfo>1 else axes[0]
            ax1.plot(loc_ref["common_grid"],loc_ref[species],'k-',label="Reference")
            ax1.plot(loc_ref["common_grid"],loc_redA[species],'r-',label="Scheme A")
            ax1.plot(loc_ref["common_grid"],loc_redB[species],'b-',label="Scheme B")
            ax1.set_ylabel(f"{species}")
            ax1.set_title(f"{t_name}")
            
            if T == 1000 : 
                ax1.set_xlim([0,0.04])
            elif T ==1200 : 
                ax1.set_xlim([0,0.0025])
            elif T == 1400 : 
                ax1.set_xlim([0,0.0007])
            if idx == n_transfo-1:
                ax1.set_xlabel("Time")
            if idx==0: ax1.legend()

            # tableau
            ax2 = axes[idx][1] if n_transfo>1 else axes[1]
            ax2.axis('off')
            cell_text = [
                [f"{loc_A_rea:.3f}", f"{loc_B_rea:.3f}", f"{loc_B_rea/loc_A_rea:.3f}"],
                [f"{loc_A_abs:.3f}", f"{loc_B_abs:.3f}",f"{loc_B_abs/loc_A_abs:.3f}"],
                [f"{loc_A_rmse:.3f}", f"{loc_B_rmse:.3f}", f"{loc_B_rmse/loc_A_rmse:.3f}"]
            ]
            col_labels = ['Scheme A','Scheme B',"Ratio B/A "]
            table = ax2.table(cellText=cell_text,
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center')
            
            ratios = [loc_B_rea/loc_A_rea,
                    loc_B_abs/loc_A_abs,
                    loc_B_rmse/loc_A_rmse
                    ]
            for r in range(len(ratios)):
                if ratios[r] > 1.0:
                    # bleu si >1
                    table[(r+1,2)].set_facecolor('#fb9a99')  #fb9a99 bleu clair
                elif ratios[r] < 1.0:
                    # rouge si <1
                    table[(r+1,2)].set_facecolor('#a6cee3')  # rouge clair
                else:
                    # neutre si =1
                    table[(r+1,2)].set_facecolor('white')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5,2.0)

        plt.tight_layout()
        plt.savefig(f"heatmap/Profile/{species}/{species}_{T}_{ER}_all_transfos.png", dpi=300)
        plt.close()