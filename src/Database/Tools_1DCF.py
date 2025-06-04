import numpy as np
import cantera as ct
import pandas as pd 
import os
from scipy.interpolate import interp1d
from .utils import Create_directory, concat_csv_list


def Sim1D_CF(t_gas,fuel1,fuel2,oxidizer,case_1D,type,dossier,save) : 
    
    dossier = Create_directory(dossier, type)
    all_df = pd.DataFrame()

    for case in case_1D:
        pressure, T_ox, strain_rate,mixture_frac = case

        # Définir le mélange de carburant (phi = 1)
        fuel_mix = f'{fuel1}:{mixture_frac}, {fuel2}:{1 - mixture_frac}'

        # Créer objets Quantity pour obtenir les densités
        F = ct.Quantity(t_gas, constant='HP')
        A = ct.Quantity(t_gas, constant='HP')
        F.TPX = 300, pressure, fuel_mix
        A.TPX = T_ox, pressure, oxidizer
        rho_f = F.density
        rho_o = A.density

        # Débits massiques à strain donné et largeur fixée
        width = 0.02  # m — typique pour contre-courant
        vf = vo = strain_rate * width / 2
        mdot_fuel = vf * rho_f
        mdot_ox = vo * rho_o

        # Préparer la flamme
        gas = t_gas  # utiliser t_gas comme modèle Cantera
        f = ct.CounterflowDiffusionFlame(gas, width=width)
        f.P = pressure
        f.fuel_inlet.T = 300
        f.fuel_inlet.X = fuel_mix
        f.fuel_inlet.mdot = mdot_fuel
        f.oxidizer_inlet.T = T_ox
        f.oxidizer_inlet.X = oxidizer
        f.oxidizer_inlet.mdot = mdot_ox
        f.energy_enabled = True

        f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        f.solve(loglevel=0, auto=True)

        # Extraire les données
        species_names = [f"Y_{name}" for name in f.gas.species_names]
        df = pd.DataFrame(f.Y.T, columns=species_names)
        df["grid"] = f.grid
        df["T"] = f.T
        df["velocity"] = f.velocity
        df["rho"] = f.density
        df["strain_rate"] = strain_rate
        df["P_Init"] = pressure
        df["T_Init"] = T_ox
        df["Mixt_Init"] = mixture_frac

        if save:
            fname = f"CF_SR{strain_rate:.1f}_T{T_ox}_P{pressure/101325:.2f}.csv"
            df.to_csv(os.path.join(dossier, fname), index=False)

        all_df = pd.concat([all_df, df], ignore_index=True)

    return all_df

def Sim1D_CF_Extinction(t_gas,fuel1,fuel2,oxidizer,case_1D,type,dossier,save) :
    tol_ss = [1.0e-6, 1.0e-9]
    tol_ts = [1.0e-6, 1.0e-9]
    width = 0.02 
    
    dossier = Create_directory(dossier, type)
    print(type)
    All_df =pd.DataFrame()
    for case in case_1D: 
        print(case)
        Case_df = pd.DataFrame()
        pressure, T_ox, strain_rate,mixture_frac = case
        # Définir le mélange de carburant (phi = 1)
        fuel_mix = f'{fuel1}:{mixture_frac}, {fuel2}:{1 - mixture_frac}' 
        
        f = ct.CounterflowDiffusionFlame(t_gas, width=width)
        f.flame.set_steady_tolerances(default=tol_ss)
        f.flame.set_transient_tolerances(default=tol_ts)
        f.P = ct.one_atm
        
        f.fuel_inlet.mdot = 0.305
        f.fuel_inlet.X = fuel_mix
        f.fuel_inlet.T = 300
        f.oxidizer_inlet.mdot = 0.1
        f.oxidizer_inlet.X = oxidizer
        f.oxidizer_inlet.T = T_ox
        f.energy_enabled = True
        
        F = ct.Quantity(t_gas, constant='HP')
        A = ct.Quantity(t_gas, constant='HP')
        F.TPX = 300, pressure, fuel_mix
        A.TPX = T_ox, pressure, oxidizer
        rho_f = F.density
        rho_o = A.density
        
        f.solve(loglevel=0, auto=True)
        Case_df = concat_flame(f,Case_df,rho_f,rho_o,width)
        # Gestion de la dicotomie pour l'extinction 
        exp_d_a = -1. / 2.
        exp_u_a = 1. / 2.
        exp_V_a = 1.
        exp_lam_a = 2.
        exp_mdot_a = 1. / 2.

        alpha = [1.]
        delta_alpha = 5.
        delta_alpha_factor = 2.
        delta_alpha_min = 0.01
        delta_T_min = 1  # K
        T_limit = T_ox
        
        n = 0
        n_last_burning = 0
        T_max = [np.max(f.T)]
        a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]
        
        
        
        while True:
            n += 1
            alpha.append(alpha[n_last_burning] + delta_alpha)
            strain_factor = alpha[-1] / alpha[n_last_burning]

            # Mise à l'échelle
            f.flame.grid *= strain_factor ** exp_d_a
            norm_grid = f.grid / (f.grid[-1] - f.grid[0])
            f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
            f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
            f.set_profile('velocity', norm_grid, f.velocity * strain_factor ** exp_u_a)
            f.set_profile('spread_rate', norm_grid, f.spread_rate * strain_factor ** exp_V_a)
            f.set_profile('lambda', norm_grid, f.L * strain_factor ** exp_lam_a)
            
            try:
                f.solve(loglevel=0,auto=True)
            except ct.CanteraError:
                delta_alpha /= delta_alpha_factor
                f = last_flame
                continue
            
            Tmax = np.max(f.T)
            Amax = np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))
            T_max.append(Tmax)
            a_max.append(Amax)
            
            if not np.isclose(Tmax, T_limit):
                n_last_burning = n
                last_flame = f
                Case_df = concat_flame(f,Case_df,rho_f,rho_o,width)
                
            elif ((T_max[-2] - T_max[-1]) < delta_T_min and delta_alpha < delta_alpha_min) or a_max[-2] > a_max[-1] :
                break
            else:
                delta_alpha /= delta_alpha_factor
                f = last_flame
                
        All_df = pd.concat([All_df,Case_df])
        All_df["mixture"] = mixture_frac
        All_df["pressure"] = pressure
        if save:
            fname = f"CF_T{T_ox}_extinction.csv"
            Case_df.to_csv(os.path.join(dossier, fname), index=False)
            
            
    return All_df



                   
def concat_flame(f,df,rho_fuel,rho_inlet,width) : 
    species_names = [f"Y_{name}" for name in f.gas.species_names]
    loc_df = pd.DataFrame(f.Y.T, columns=species_names)
    loc_df["grid"] =f.grid
    loc_df["T"] =f.T
    loc_df["T_Init"] = f.oxidizer_inlet.T
    loc_df["local_strain"] = np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))
    loc_df["global_strain"] = (f.fuel_inlet.mdot / rho_fuel + f.oxidizer_inlet.mdot / rho_inlet) / width
    loc_df["velocity"] = f.velocity
    return pd.concat([df,loc_df],ignore_index=True)  



def Launch_processing_1D_CF_csv(
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
    
    Data_Ref = Processing_1D_CF_ref(Data_Ref,cases,name_ref,Path,save_csv)
    Data = Processing_1D_CF_data(Data,Data_Ref,cases,name_data,Path,save_csv)
    
    return Data_Ref,Data

def Processing_1D_CF_ref(
    input_data: pd.DataFrame,
    cases: list,
    name_ref: str,
    Path: str,
    save_csv: bool
) : 
    data_processing_ref = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    for T_Init in input_data["T_Init"].unique() : 
        loc = pd.DataFrame() 
        loc_T_data_ref = input_data[input_data["T_Init"]==T_Init]
        for global_strain in loc_T_data_ref["global_strain"].unique() : 
            New_data = pd.DataFrame()
            loc_ST_T_data_ref =  loc_T_data_ref[loc_T_data_ref["global_strain"]==global_strain]
            New_data["common_grid"] = loc_ST_T_data_ref["grid"]
            for s in species : 
                int_func = interp1d(loc_ST_T_data_ref["grid"],loc_ST_T_data_ref[s],fill_value='extrapolate')
                New_data[s] = int_func(New_data["common_grid"])
            
                int_func = interp1d(loc_ST_T_data_ref["grid"],loc_ST_T_data_ref["T"],fill_value='extrapolate')
                
            New_data["T"] = int_func(New_data["common_grid"])
            
            int_func = interp1d(loc_ST_T_data_ref["grid"], loc_ST_T_data_ref["velocity"], fill_value="extrapolate")
            New_data["velocity"]=int_func(New_data["common_grid"])
            
            New_data["T_Init"] = T_Init
            New_data["global_strain"] = global_strain
            New_data["local_strain"] = loc_ST_T_data_ref["local_strain"].iloc[0]
            
            loc = pd.concat([loc,New_data])
        
        data_processing_ref = pd.concat([loc,data_processing_ref])
    data_processing_ref["max_strain"] = data_processing_ref["local_strain"].max()
            
    if save_csv : 
        data_processing_ref.to_csv(os.path.join(Path, f"Processing_{name_ref}.csv"))
        
    return data_processing_ref


def Processing_1D_CF_data(
    input_data: pd.DataFrame,
    input_data_ref : pd.DataFrame,
    cases: list,
    name_data: str,
    Path: str,
    save_csv: bool
) : 
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]

    for T_Init in input_data["T_Init"].unique() : 
        loc = pd.DataFrame()
        loc_data_processing_ref_T = input_data_ref[input_data_ref["T_Init"]==T_Init]
        loc_data_T=input_data[input_data["T_Init"]==T_Init] 
        for global_strain in loc_data_T["global_strain"].unique() : 
            if global_strain not in loc_data_processing_ref_T["global_strain"].unique() :
                
                break
            else : 
                New_data = pd.DataFrame()
                data_loc = loc_data_T[loc_data_T["global_strain"]==global_strain]
                New_data["common_grid"] =loc_data_processing_ref_T[loc_data_processing_ref_T["global_strain"]==global_strain]["common_grid"] 
                for s in species:
                    int_func = interp1d(data_loc["grid"], data_loc[s], fill_value="extrapolate")
                    New_data[s] = int_func(New_data["common_grid"])
                    
                int_func = interp1d(data_loc["grid"], data_loc["T"], fill_value="extrapolate")
                New_data["T"] = int_func(New_data["common_grid"])
                
                int_func = interp1d(data_loc["grid"], data_loc["velocity"], fill_value="extrapolate")
                New_data["velocity"]=int_func(New_data["common_grid"])
                
                New_data["T_Init"] = T_Init
                New_data["global_strain"] = global_strain
                New_data["local_strain"] = data_loc["local_strain"].iloc[0]
                
                loc = pd.concat([loc,New_data])
                
        loc["max_strain"] = max(loc_data_T["local_strain"])
        data_processing = pd.concat([data_processing,loc])
    
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_data}.csv"))
        
    return data_processing 

