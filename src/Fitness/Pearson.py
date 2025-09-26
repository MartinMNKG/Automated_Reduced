import numpy as np 
import pandas as pd 
from scipy import stats

def Pearson(D, R, species_cols, flag_output=True):

    results = []

    # Boucle sur chaque condition expérimentale
    for (phi, T) in D.groupby(["Phi_Init", "T_Init"]).groups.keys():
        # Sous-ensembles
        D_sub = D[(D["Phi_Init"] == phi) & (D["T_Init"] == T)].sort_values("common_grid")
        R_sub = R[(R["Phi_Init"] == phi) & (R["T_Init"] == T)].sort_values("common_grid")

        # Vérification des grilles
        if not np.allclose(D_sub["common_grid"].values, R_sub["common_grid"].values):
            raise ValueError(f"Grilles temporelles différentes pour Phi={phi}, T={T}")

        # Calcul des corrélations
        for sp in species_cols:
            r, pval = stats.pearsonr(D_sub[sp].values, R_sub[sp].values)
            results.append({
                "Phi_Init": phi,
                "T_Init": T,
                "Species": sp,
                "Pearson_r": r,
                "pval": pval
            })

    df = pd.DataFrame(results)

    if flag_output:
        # tableau pivoté par Phi/T et colonnes d'espèces
        return df.pivot(index=["Phi_Init", "T_Init"], columns="Species", values="Pearson_r").reset_index()
    else:
        # Moyenne globale des corrélations
        return df["Pearson_r"].sum()

def Pearson_global(D, R, species_cols, flag_output=True):
    results = []
    for sp in species_cols:
        r, p = stats.pearsonr(D[sp].values, R[sp].values)
        results.append({"Species": sp, "r": r, "p-value": p})

    df = pd.DataFrame(results)

    if flag_output:
        return df
    else:
        return df["r"].sum()
    