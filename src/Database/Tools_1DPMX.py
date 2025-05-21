import numpy as np
import cantera as ct
import pandas as pd 
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .utils import Create_directory, concat_csv_list





def Sim1D(t_gas,fuel1,fuel2,oxidizer,case_1D,type,dossier,save) : 
    print(type)
    dossier = Create_directory(dossier,type)
    all_df = pd.DataFrame()
    for case in case_1D :
        print(case) 
        pressure, temperature, equivalence_ratio,mixture = case
        fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'

        t_gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
        t_gas.TP = temperature,pressure
        
        width = 0.05
        
        f = ct.FreeFlame(t_gas,width=width)


        # Rafinement de la flamme
        f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        f.inlet.T = temperature
        f.inlet.Y = t_gas.Y
        
        
        # Résoudre la flamme
        f.solve(loglevel=0, auto=True)
    
    
        species_names = [f"Y_{name}" for name in f.gas.species_names]
        df = pd.DataFrame(f.Y.T, columns=species_names)
        df["grid"] = f.grid
        df["T"] = f.T
        df["velocity"] = f.velocity  # velocity
        df["rho"] = f.density

        # Add metadata
        df["P_Init"] = pressure
        df["T_Init"] = temperature
        df["Phi_Init"] = equivalence_ratio
        df["Mixt_Init"] = mixture
        if save : 
            df.to_csv(f"{dossier}/1D_PMX_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv", index=False)

        all_df = pd.concat([all_df,df],ignore_index=True)
    return all_df




def Launch_processing_1D_PMX_csv(
    Ref_csv : list, 
    Data_csv : list, 
    cases : list, 
    name_ref : str, 
    name_data : str, 
    Path : str, 
    save_csv : bool
) : 
    Data_Ref = concat_csv_list(Ref_csv)
    Data = concat_csv_list(Data_csv)
    
    Data_Ref = Processing_1D_PMX_ref(Data_Ref,cases,name_ref,Path,save_csv)
    Data = Processing_1D_PMX_data(Data,Data_Ref,cases,name_data,Path,save_csv)
    
    return Data_Ref,Data

def Processing_1D_PMX_ref(
    input_data: pd.DataFrame,
    cases: list,
    name_ref: str,
    Path: str,
    save_csv: bool
) : 
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    
    for c in cases : 
        New_data_ref = pd.DataFrame()
        pressure, temperature, equivalence_ratio, mixture = c 
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["Phi_Init"] == equivalence_ratio)&(input_data["Mixt_Init"] == mixture)].copy()
        data_loc["grid"] = shift_1D(data_loc["grid"],data_loc["T"])
        New_data_ref["common_grid"] = data_loc["grid"]
        
        for s in species : 
            int_func = interp1d(data_loc["grid"],data_loc[s],fill_value='extrapolate')
            New_data_ref[s] = int_func(New_data_ref["common_grid"])
        
        int_func = interp1d(data_loc["grid"],data_loc["T"],fill_value='extrapolate')
        New_data_ref["T"] = int_func(New_data_ref["common_grid"])
        
        New_data_ref["velocity"] = data_loc["velocity"].iloc[0]
        New_data_ref["P_Init"]  = pressure
        New_data_ref["T_Init"]  = temperature
        New_data_ref["Phi_Init"]  =   equivalence_ratio  
        New_data_ref["Mixt_Init"] = mixture     
        
        data_processing = pd.concat([data_processing,New_data_ref],ignore_index=True) 
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_ref}.csv"))
        
    return data_processing 

def Processing_1D_PMX_data(
    input_data: pd.DataFrame,
    input_data_ref : pd.DataFrame,
    cases: list,
    name_data: str,
    Path: str,
    save_csv: bool
) : 
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    
    for c in cases : 
        pressure, temperature, equivalence_ratio, mixture = c
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["Phi_Init"] == equivalence_ratio)&(input_data["Mixt_Init"] == mixture)].copy()
        loc_common_grid = input_data_ref[(input_data_ref["P_Init"] == pressure)&(input_data_ref["T_Init"] ==temperature)&(input_data_ref["Phi_Init"] == equivalence_ratio)&(input_data_ref["Mixt_Init"] == mixture)]["common_grid"]
        
        New_data = pd.DataFrame()

        data_loc["grid"] = shift_1D(data_loc["grid"],data_loc["T"])
        New_data["common_grid"] = loc_common_grid
        
        for s in species:
            int_func = interp1d(data_loc["grid"], data_loc[s], fill_value="extrapolate")
            New_data[s] = int_func(loc_common_grid)
            
        int_func = interp1d(data_loc["grid"], data_loc["T"], fill_value="extrapolate")
        New_data["T"] = int_func(loc_common_grid)
        
        New_data["velocity"] = data_loc["velocity"].iloc[0]
        New_data["P_Init"] = pressure
        New_data["T_Init"] = temperature
        New_data["Phi_Init"] = equivalence_ratio
        New_data["Mixt_Init"] = mixture
            
        data_processing = pd.concat([data_processing,New_data],ignore_index=True) 
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_data}.csv"))
        
    return data_processing 
    
    

def shift_1D(grid: list, T: list) -> list:
    gradient = np.gradient(T, grid)
    indice_gradient = np.argmax(gradient)
    shift_grid = grid - grid.iloc[indice_gradient]

    return shift_grid
