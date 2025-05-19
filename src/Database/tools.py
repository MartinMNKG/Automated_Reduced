import numpy as np
import cantera as ct
import pandas as pd 
import os
import re
import sys
import time
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from sklearn.preprocessing import StandardScaler
import glob
import seaborn as sns

class MinMaxScaler:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)

    def transform(self, x):
        x = (x - self.min) / (self.max - self.min + 1e-7)
        return x

    def inverse_transform(self, x):
        x = self.min + x * (self.max - self.min + 1e-7)
        return x
    
def generate_test_cases_bifuel(pressure_range,temp_range, second_param,mixture):
    test_cases = list(itertools.product(pressure_range, temp_range, second_param,mixture))

    # Convertir les pressions en Pascals (Pa) car Cantera utilise les Pascals
    test_cases = [(p * 101325, T, second,mixture) for p, T, second,mixture in test_cases]
    
    return test_cases

def Create_directory(main_path,name) : 
    
    dossier = os.path.join(main_path,f"{name}")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass
    return dossier

def Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case_0D,dt,tmax,type,dossier,save) : 
    if save == True :
        dossier = Create_directory(dossier,type)
    all_df = pd.DataFrame()
    for case in case_0D :
        pressure, temperature, equivalence_ratio,mixture = case
        fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'
        t_gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
        t_gas.TP = temperature,pressure
        
        r = ct.IdealGasConstPressureReactor(t_gas)
        sim = ct.ReactorNet([r])
        y_list = []
        y_list.append(r.Y)
        
        time = 0
        
        gas_eq.TP = temperature,pressure
        gas_eq.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
        gas_eq.equilibrate("HP")

        states = ct.SolutionArray(t_gas, extra=["t"])
        
        while time <tmax:
            time +=dt
            sim.advance(time)
            states.append(r.thermo.state, t=time)           
            
        df_states = states.to_pandas()
        df_states["P_Init"] = pressure
        df_states["T_Init"] = temperature
        df_states["Phi_Init"] = equivalence_ratio
        df_states["Mixt_Init"] = mixture
        if save == True:   
            df_states.to_csv(f"{dossier}/0Dreactor_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv")
        
        all_df = pd.concat([all_df,df_states],ignore_index=True)
    return all_df

def Sim1D(t_gas,fuel1,fuel2,oxidizer,case_1D,type,dossier,save) : 
    dossier = Create_directory(dossier,type)
    all_df = pd.DataFrame()
    for case in case_1D : 
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
    
    
    for case in case_1D: 
        All_df = pd.DataFrame()
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
        All_df = concat_flame(f,All_df,rho_f,rho_o,width)
        # Gestion de la dicotomie pour l'extinction 
        exp_d_a = -1. / 2.
        exp_u_a = 1. / 2.
        exp_V_a = 1.
        exp_lam_a = 2.
        exp_mdot_a = 1. / 2.

        alpha = [1.]
        delta_alpha = 10.
        delta_alpha_factor = 10.
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
                All_df = concat_flame(f,All_df,rho_f,rho_o,width)
                
            elif ((T_max[-2] - T_max[-1]) < delta_T_min and delta_alpha < delta_alpha_min) or a_max[-2] > a_max[-1]:
                All_df = concat_flame(f,All_df,rho_f,rho_o,width)
                break
            else:
                delta_alpha /= delta_alpha_factor
                f = last_flame
        
        if save:
            fname = f"CF_T{T_ox}_extinction.csv"
            All_df.to_csv(os.path.join(dossier, fname), index=False)
            
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

def Launch_processing_0D_csv(
    Ref_csv : list,
    Data_csv: list, 
    cases : list, 
    length: int, 
    name_ref : str, 
    name_data :str,
    Path : str,
    save_csv : bool 
) : 
    Data_Ref = concat_csv_list(Ref_csv)
    Data = concat_csv_list(Data_csv)
    
    Data_Ref = Processing_0D_ref(Data_Ref,cases,length,name_ref,Path,save_csv)
    Data = Processing_0D_data(Data,Data_Ref,cases,name_data,Path,save_csv)
    
    return Data_Ref,Data

def Processing_0D_ref(
    input_data: pd.DataFrame,
    cases: list,
    length: int,
    name_ref: str,
    Path: str,
    save_csv: bool
):
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    
    for c in cases : 
        New_data_ref = pd.DataFrame()
        
        pressure, temperature, equivalence_ratio, mixture = c 
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["Phi_Init"] == equivalence_ratio)&(input_data["Mixt_Init"] == mixture)].copy()
        IDT_d = Calc_ai_delay(data_loc["t"], data_loc["T"])
        
        
        ind = find_convergence_index(data_loc["T"])
        New_data_ref["common_grid"] = Generate_common_grid(
            data_loc["t"].iloc[:ind], data_loc["T"].iloc[:ind], length
        )

        for s in species:
            int_func = interp1d(data_loc["t"], data_loc[s], fill_value="extrapolate")
            New_data_ref[s] = int_func(New_data_ref["common_grid"])

        int_func = interp1d(data_loc["t"], data_loc["T"], fill_value="extrapolate")
        New_data_ref["T"] = int_func(New_data_ref["common_grid"])

        New_data_ref["IDT"] = IDT_d
        New_data_ref["P_Init"] = pressure
        New_data_ref["T_Init"] = temperature
        New_data_ref["Phi_Init"] = equivalence_ratio
        New_data_ref["Mixt_Init"] = mixture
        data_processing = pd.concat([data_processing, New_data_ref], ignore_index=True)
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_ref}.csv"))
        
    return data_processing

def Processing_0D_data(
    input_data: pd.DataFrame,
    input_data_ref : pd.DataFrame,
    cases: list,
    name_data: str,
    Path: str,
    save_csv: bool
):
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    
    for c in cases : 
        pressure, temperature, equivalence_ratio, mixture = c
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["Phi_Init"] == equivalence_ratio)&(input_data["Mixt_Init"] == mixture)].copy()
        
        # get common grid 
        loc_common_grid = input_data_ref[(input_data_ref["P_Init"] == pressure)&(input_data_ref["T_Init"] ==temperature)&(input_data_ref["Phi_Init"] == equivalence_ratio)&(input_data_ref["Mixt_Init"] == mixture)]["common_grid"]
        
        New_data = pd.DataFrame()
        
        IDT_r = Calc_ai_delay(data_loc["t"], data_loc["T"])

        New_data["common_grid"] = loc_common_grid

        for s in species:
            int_func = interp1d(data_loc["t"], data_loc[s], fill_value="extrapolate")
            New_data[s] = int_func(loc_common_grid)

        int_func = interp1d(data_loc["t"], data_loc["T"], fill_value="extrapolate")
        New_data["T"] = int_func(loc_common_grid)

        New_data["IDT"] = IDT_r
        New_data["P_Init"] = pressure
        New_data["T_Init"] = temperature
        New_data["Phi_Init"] = equivalence_ratio
        New_data["Mixt_Init"] = mixture
        data_processing = pd.concat([data_processing, New_data], ignore_index=True)
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_data}.csv"))
    
    return data_processing

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
    data_processing = pd.DataFrame()
    species = [col for col in input_data.columns if col.startswith("Y_")]
    
    for c in cases : 
        New_data_ref = pd.DataFrame()
        pressure, temperature, strain_rate, mixture = c 
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["strain_rate"] == strain_rate)&(input_data["Mixt_Init"] == mixture)].copy()
        New_data_ref["common_grid"] = data_loc["grid"]
        
        for s in species : 
            int_func = interp1d(data_loc["grid"],data_loc[s],fill_value='extrapolate')
            New_data_ref[s] = int_func(New_data_ref["common_grid"])
        
        int_func = interp1d(data_loc["grid"],data_loc["T"],fill_value='extrapolate')
        New_data_ref["T"] = int_func(New_data_ref["common_grid"])
        
        New_data_ref["velocity"] = data_loc["velocity"].iloc[0]
        New_data_ref["P_Init"]  = pressure
        New_data_ref["T_Init"]  = temperature
        New_data_ref["strain_rate"]  =   strain_rate  
        New_data_ref["Mixt_Init"] = mixture     
        
        data_processing = pd.concat([data_processing,New_data_ref],ignore_index=True) 
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_ref}.csv"))
        
    return data_processing


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
    
    for c in cases : 
        pressure, temperature, strain_rate, mixture = c
        data_loc= input_data[(input_data["P_Init"] == pressure)&(input_data["T_Init"] ==temperature)&(input_data["strain_rate"] == strain_rate)&(input_data["Mixt_Init"] == mixture)].copy()
        loc_common_grid = input_data_ref[(input_data_ref["P_Init"] == pressure)&(input_data_ref["T_Init"] ==temperature)&(input_data_ref["strain_rate"] == strain_rate)&(input_data_ref["Mixt_Init"] == mixture)]["common_grid"]
        
        New_data = pd.DataFrame()
        New_data["common_grid"] = loc_common_grid
        
        for s in species:
            int_func = interp1d(data_loc["grid"], data_loc[s], fill_value="extrapolate")
            New_data[s] = int_func(loc_common_grid)
            
        int_func = interp1d(data_loc["grid"], data_loc["T"], fill_value="extrapolate")
        New_data["T"] = int_func(loc_common_grid)
        
        New_data["velocity"] = data_loc["velocity"].iloc[0]
        New_data["P_Init"] = pressure
        New_data["T_Init"] = temperature
        New_data["strain_rate"] = strain_rate
        New_data["Mixt_Init"] = mixture
            
        data_processing = pd.concat([data_processing,New_data],ignore_index=True) 
        
    if save_csv : 
        data_processing.to_csv(os.path.join(Path, f"Processing_{name_data}.csv"))
        
    return data_processing 

def Calc_ai_delay(time,temp) :
    loc_time = time.reset_index(drop=True)
    loc_temp = temp.reset_index(drop=True)
    diff_temp = np.diff(loc_temp)/np.diff(loc_time)
    ign_pos = np.argmax(diff_temp)
    output=loc_time[ign_pos]
    return output

def shift_1D(grid: list, T: list) -> list:
    gradient = np.gradient(T, grid)
    indice_gradient = np.argmax(gradient)
    shift_grid = grid - grid.iloc[indice_gradient]

    return shift_grid

def Generate_common_grid(grid,temp,lenght) :
    
    gradient = np.abs(np.gradient(temp,grid))
    density =  gradient / np.trapz(gradient,grid)
    F = cumtrapz(density,grid,initial=0)
    inv_cdf = interp1d( F,grid)
    x_vals = np.linspace(0,F[-1],lenght)
    return inv_cdf(x_vals)
    # return np.logspace(min(time),max(time),lenght)

def find_convergence_index(series: pd.Series, window: int = 5, tolerance: float = 1e-2) -> int:
    final_value = series.iloc[-1]
    for i in range(len(series) - window):
        window_values = series.iloc[i:i + window]
        if np.all(np.abs(window_values - final_value) <= tolerance):
            return i + window  # Return the index where convergence ends
    return len(series)  # No convergence detected

def extract_values(path):
    # Regex améliorée : on impose que P soit suivi de chiffres/point AVANT ".csv"
    match = re.search(r'ER([0-9.]+)_T([0-9.]+)_P([0-9.]+)\.csv$', path)
    if match:
        try:
            er = float(match.group(1))
            temp = float(match.group(2))
            press = float(match.group(3))
            return (er, temp, press)
        except ValueError:
            pass  # en cas de float invalide
    # Cas d'erreur ou fichier mal nommé
    print(f"⚠️ Mauvais format de fichier : {path}")
    return (float('inf'), float('inf'), float('inf'))

def concat_csv_list(csv_paths: list[str]) -> pd.DataFrame:
    """
    Concatène une liste de fichiers CSV en un seul DataFrame pandas.

    Args:
        csv_paths (list[str]): Liste des chemins vers les fichiers CSV.

    Returns:
        pd.DataFrame: DataFrame concaténé contenant les données de tous les fichiers.
    """
    all_data = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            all_data.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {path}: {e}")

    return pd.concat(all_data, ignore_index=True)