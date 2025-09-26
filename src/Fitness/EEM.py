import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _preprocess_data(D, R, species_cols, do_log=False, norm_type=None):
    """Prépare les données : log et/ou normalisation globale.
    Fit du scaler sur D (detailed) et transform sur D et R."""
    
    Dn = D.copy()
    Rn = R.copy()
    # 1️⃣ log si demandé
    if do_log==True:
      
        Dn[species_cols] = np.log(np.maximum(Dn[species_cols], 1e-12))
        Rn[species_cols] = np.log(np.maximum(Rn[species_cols], 1e-12))
    # 2️⃣ normalisation globale si demandé
    if norm_type is not None:
       
        if norm_type == "minmax":
            
            scaler = MinMaxScaler()
        elif norm_type =="standard" :
            
            scaler = StandardScaler()
        scaler.fit(Dn[species_cols])  # fit sur D seulement
        Dn[species_cols] = scaler.transform(Dn[species_cols])
        Rn[species_cols] = scaler.transform(Rn[species_cols])
    return Dn, Rn


def RE(D,R,species_cols,flag_output,do_log=False,norm_type=None):
    results_true, results_false = [], []
    Dn, Rn = _preprocess_data(D,R,species_cols,do_log,norm_type)

    for (phi, T) in Dn.groupby(["Phi_Init","T_Init"]).groups.keys():
        D_sub = Dn[(Dn["Phi_Init"]==phi)&(Dn["T_Init"]==T)].sort_values("common_grid")
        R_sub = Rn[(Rn["Phi_Init"]==phi)&(Rn["T_Init"]==T)].sort_values("common_grid")
        if not np.allclose(D_sub["common_grid"].values,R_sub["common_grid"].values):
            raise ValueError(f"Grilles différentes pour Phi={phi}, T={T}")
        for sp in species_cols:
            diff = np.abs(D_sub[sp].values - R_sub[sp].values)
            err = diff / np.maximum(np.abs(D_sub[sp]).values,1e-12)
            results_true.append({"Phi_Init":phi,"T_Init":T,"Species":sp,"RE":np.sum(err)})
            results_false.append(np.sum(err))
    df = pd.DataFrame(results_true)
    return df.pivot(index=["Phi_Init","T_Init"],columns="Species",values="RE").reset_index() if flag_output else np.sum(results_false)

def ABS(D,R,species_cols,flag_output,do_log=False,norm_type=None):
    result_true, results_false = [], []
    Dn, Rn = _preprocess_data(D,R,species_cols,do_log,norm_type)

    for (phi, T) in Dn.groupby(["Phi_Init","T_Init"]).groups.keys():
        D_sub = Dn[(Dn["Phi_Init"]==phi)&(Dn["T_Init"]==T)].sort_values("common_grid")
        R_sub = Rn[(Rn["Phi_Init"]==phi)&(Rn["T_Init"]==T)].sort_values("common_grid")
        if not np.allclose(D_sub["common_grid"].values,R_sub["common_grid"].values):
            raise ValueError(f"Grilles différentes pour Phi={phi}, T={T}")
        for sp in species_cols:
            err = np.abs(D_sub[sp].values - R_sub[sp].values)
            result_true.append({"Phi_Init":phi,"T_Init":T,"Species":sp,"ABS":np.sum(err)})
            results_false.append(np.sum(err))
    df = pd.DataFrame(result_true)
    return df.pivot(index=["Phi_Init","T_Init"],columns="Species",values="ABS").reset_index() if flag_output else np.sum(results_false)

def RMSE(D,R,species_cols,flag_output,do_log=False,norm_type=None):
    results_true, results_false = [], []
    Dn, Rn = _preprocess_data(D,R,species_cols,do_log,norm_type)

    for (phi, T) in Dn.groupby(["Phi_Init","T_Init"]).groups.keys():
        D_sub = Dn[(Dn["Phi_Init"]==phi)&(Dn["T_Init"]==T)].sort_values("common_grid")
        R_sub = Rn[(Rn["Phi_Init"]==phi)&(Rn["T_Init"]==T)].sort_values("common_grid")
        if not np.allclose(D_sub["common_grid"].values,R_sub["common_grid"].values):
            raise ValueError(f"Grilles différentes pour Phi={phi}, T={T}")
        for sp in species_cols:
            diff = D_sub[sp].values - R_sub[sp].values
            rmse = np.sqrt(np.mean(diff**2))
            results_true.append({"Phi_Init":phi,"T_Init":T,"Species":sp,"RMSE":rmse})
            results_false.append(rmse)
    df = pd.DataFrame(results_true)
    return df.pivot(index=["Phi_Init","T_Init"],columns="Species",values="RMSE").reset_index() if flag_output else np.sum(results_false)

def IE(D,R,species_cols,flag_output,do_log=False,norm_type=None):
    results_true, results_false = [], []
    Dn, Rn = _preprocess_data(D,R,species_cols,do_log,norm_type)

    for (phi, T) in Dn.groupby(["Phi_Init","T_Init"]).groups.keys():
        D_sub = Dn[(Dn["Phi_Init"]==phi)&(Dn["T_Init"]==T)].sort_values("common_grid")
        R_sub = Rn[(Rn["Phi_Init"]==phi)&(Rn["T_Init"]==T)].sort_values("common_grid")
        t = D_sub["common_grid"].values
        if not np.allclose(t,R_sub["common_grid"].values):
            raise ValueError(f"Grilles différentes pour Phi={phi}, T={T}")
        for sp in species_cols:
            diff = np.abs(R_sub[sp].values - D_sub[sp].values)
            err = np.trapezoid(diff,x=t)
            results_true.append({"Phi_Init":phi,"T_Init":T,"Species":sp,"IE":err})
            results_false.append(err)
    df = pd.DataFrame(results_true)
    return df.pivot(index=["Phi_Init","T_Init"],columns="Species",values="IE").reset_index() if flag_output else np.sum(results_false)

def RE_A(D,R,species_cols,flag_output,do_log=False,norm_type=None):
    results_true, results_false = [], []
    Dn, Rn = _preprocess_data(D,R,species_cols,do_log,norm_type)

    for (phi, T) in Dn.groupby(["Phi_Init","T_Init"]).groups.keys():
        D_sub = Dn[(Dn["Phi_Init"]==phi)&(Dn["T_Init"]==T)].sort_values("common_grid")
        R_sub = Rn[(Rn["Phi_Init"]==phi)&(Rn["T_Init"]==T)].sort_values("common_grid")
        t = D_sub["common_grid"].values
        if not np.allclose(t,R_sub["common_grid"].values):
            raise ValueError(f"Grilles différentes pour Phi={phi}, T={T}")
        for sp in species_cols:
            a_d = np.trapezoid(D_sub[sp].values,x=t)
            a_r = np.trapezoid(R_sub[sp].values,x=t)
            err = np.abs(a_d-a_r)/np.maximum(np.abs(a_d),1e-12)
            results_true.append({"Phi_Init":phi,"T_Init":T,"Species":sp,"REAREA":err})
            results_false.append(err)
    df = pd.DataFrame(results_true)
    return df.pivot(index=["Phi_Init","T_Init"],columns="Species",values="REAREA").reset_index() if flag_output else np.sum(results_false)
