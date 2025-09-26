import numpy as np
import cantera as ct
import pandas as pd 
import os
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from .utils import Create_directory, concat_csv_list



def Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case_0D,dt,tmax,type,dossier,save) :
    # print(type) 
    if save == True :
        dossier = Create_directory(dossier,type)
    all_df = pd.DataFrame()
    # print("P,T,Phi,Mix")
    for case in case_0D :
        # print(f"case :{case} ")
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
        
        # POUR NH3 H2 ARTICLE MARTIN 
        if temperature== 1000 : 
            tmax = 0.06
        elif temperature == 1200 :
            tmax = 0.005
        elif temperature == 1400 : 
            tmax = 0.0015
            
            
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
        # New_data_ref["common_grid"] = Generate_common_grid(
        #     data_loc["t"].iloc[:ind], data_loc["T"].iloc[:ind], length
        # )
        New_data_ref["common_grid"] = np.linspace(min(data_loc["t"]), max(data_loc["t"]), length)

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


def Calc_ai_delay(time,temp) :
    loc_time = time.reset_index(drop=True)
    loc_temp = temp.reset_index(drop=True)
    diff_temp = np.diff(loc_temp)/np.diff(loc_time)
    ign_pos = np.argmax(diff_temp)
    output=loc_time[ign_pos]
    return output