import numpy as np
import cantera as ct
import pandas as pd 
import os
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

def Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case_0D,dt,tmax,type,dossier) : 
    dossier = Create_directory(dossier,type)
    all_df = pd.DataFrame()
    for case in case_0D :
        pressure, temperature, equivalence_ratio,mixture = case
        fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'
        t_gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
        t_gas.TP = temperature,pressure
        
        r = ct.IdealGasConstPressureReactor(t_gas)
        sim = ct.ReactorNet([r])
        t_list = [0]
        temp_list = [t_gas.T]
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
            
        states.save(
            f"{dossier}/0Dreactor_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv",overwrite=True
        )      

def Sim1D(t_gas,fuel1,fuel2,oxidizer,case_1D,type,dossier) : 
    dossier = Create_directory(dossier,type)
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
        
        f.save(f"{dossier}/1D_PMX_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv",overwrite=True)

def Processing_0D(list_csv_d,list_csv_r,cases,time_shift,log,scaler,lenght,name_d,name_r,Path) : 

    all_data_d = pd.DataFrame()
    all_data_r = pd.DataFrame()
    
    list_csv = zip(list_csv_d,list_csv_r)
    
    for csv_d,csv_r in list_csv : 
        
        pressure, temperature, equivalence_ratio,mixture = cases[list_csv_d.index(csv_d)]
        New_data_d = pd.DataFrame()
        data_d=pd.read_csv(csv_d)
        data_r=pd.read_csv(csv_r)
        species = [col for col in data_r.columns if col.startswith("Y_")]
        IDT_d = Calc_ai_delay(data_d["t"],data_d["T"])
        if time_shift == True : 
            data_d["t"] = data_d["t"] - IDT_d
        
        if log == True : 
            data_d[species]=data_d[species].apply(np.log)  
        
        ind = find_convergence_index(data_d["T"])
        New_data_d["common_grid"] = Generate_common_grid(data_d["t"].iloc[:ind],data_d["T"].iloc[:ind],lenght)
        
        for s in species : 
            int_func = interp1d(data_d["t"],data_d[s],fill_value='extrapolate')
            New_data_d[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_d["t"],data_d["T"],fill_value='extrapolate')
        New_data_d["T"] = int_func(New_data_d["common_grid"])
        
        if scaler== True : 
            
            # scl = MinMaxScaler()
            scl = StandardScaler()
            scl.fit(New_data_d[species])
            New_data_d[species] = scl.transform(New_data_d[species])

    
        
        
        New_data_d["IDT"] = IDT_d
        New_data_d["P_Init"]  = pressure
        New_data_d["T_Init"]  = temperature
        New_data_d["Phi_Init"]  =   equivalence_ratio  
        New_data_d["Mixt_Init"] = mixture           
        
        all_data_d = pd.concat([all_data_d,New_data_d],ignore_index=True)
    
        
        New_data_r = pd.DataFrame()
        
        IDT_r = Calc_ai_delay(data_r["t"],data_r["T"])
        if time_shift == True : 
            data_r["t"] = data_r["t"] - IDT_r
        if log == True : 
            data_r[species]=data_r[species].apply(np.log)  
        
        New_data_r["common_grid"] = New_data_d["common_grid"]
        
        for s in species : 
            int_func = interp1d(data_r["t"],data_r[s],fill_value='extrapolate')
            New_data_r[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_r["t"],data_r["T"],fill_value='extrapolate')
        New_data_r["T"] = int_func(New_data_d["common_grid"])
            
        if scaler== True : 
            New_data_r[species] = scl.transform(New_data_r[species])
        
        
        
        New_data_r["IDT"] = IDT_r
        New_data_r["P_Init"]  = pressure
        New_data_r["T_Init"]  = temperature
        New_data_r["Phi_Init"]  =   equivalence_ratio  
        New_data_r["Mixt_Init"] = mixture  
    
        all_data_r = pd.concat([all_data_r,New_data_r],ignore_index=True)     
    
    all_data_d.to_csv(os.path.join(Path,f"Processing_{name_d}.csv"))
    all_data_r.to_csv(os.path.join(Path,f"Processing_{name_r}.csv"))
    
    
    return all_data_d, all_data_r
 
def Processing_1D(list_csv_d,list_csv_r,cases,grid_shift,log,scaler,lenght,name_d,name_r,Path)  : 
    
    all_data_d = pd.DataFrame()
    all_data_r = pd.DataFrame()
    
    list_csv = zip(list_csv_d,list_csv_r)
    for csv_d,csv_r in list_csv : 
    
        pressure, temperature, equivalence_ratio,mixture = cases[list_csv_d.index(csv_d)]
        New_data_d = pd.DataFrame()
        data_d=pd.read_csv(csv_d)
        species_d = [col for col in data_d.columns if col.startswith("Y_")]
        data_d_shift_1D = shift_1D(data_d["grid"],data_d["T"])
        
        if grid_shift == True : 
            data_d["grid"] = data_d["grid"] - data_d_shift_1D
        
        if log == True : 
            data_d[species_d]=data_d[species_d].apply(np.log)  
        
        # ind = find_convergence_index(data_d["T"])
        # New_data_d["common_grid"] = Generate_common_grid(data_d["grid"].iloc[:ind],data_d["T"].iloc[:ind],lenght)
        New_data_d["common_grid"] = data_d["grid"]
        
        for s in species_d : 
            int_func = interp1d(data_d["grid"],data_d[s],fill_value='extrapolate')
            New_data_d[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_d["grid"],data_d["T"],fill_value='extrapolate')
        New_data_d["T"] = int_func(New_data_d["common_grid"])
        
        all_scl =[]
        if scaler== True : 
            for s in species_d : 
                scl = MinMaxScaler()
                scl.fit(New_data_d[s])
                New_data_d[s] = scl.transform(New_data_d[s])
                all_scl.append(scl)
                
       
        
        New_data_d["velocity"] = data_d["velocity"][0]
        New_data_d["P_Init"]  = pressure
        New_data_d["T_Init"]  = temperature
        New_data_d["Phi_Init"]  =   equivalence_ratio  
        New_data_d["Mixt_Init"] = mixture     
        
        all_data_d = pd.concat([all_data_d,New_data_d],ignore_index=True) 
        
        
        New_data_r = pd.DataFrame()
        data_r=pd.read_csv(csv_r)
        species_r = [col for col in data_r.columns if col.startswith("Y_")]
        data_r_shift_1D = shift_1D(data_r["grid"],data_r["T"])
        
        if grid_shift == True : 
            data_r["grid"] = data_r["grid"] - data_r_shift_1D
            
        if log == True : 
            data_r[species_r]=data_r[species_r].apply(np.log)  
        
        New_data_r["common_grid"] = New_data_d["common_grid"]
        
        for s in species_r : 
            int_func = interp1d(data_r["grid"],data_r[s],fill_value='extrapolate')
            New_data_r[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_r["grid"],data_r["T"],fill_value='extrapolate')
        New_data_r["T"] = int_func(New_data_d["common_grid"])
        
        if scaler== True : 
            for s in species_r : 
                scl = all_scl[species_d.index(s)]
                New_data_r[s] = scl.transform(New_data_r[s])
        
        
        
        New_data_r["IDT"] = data_r["velocity"][0]
        New_data_r["P_Init"]  = pressure
        New_data_r["T_Init"]  = temperature
        New_data_r["Phi_Init"]  =   equivalence_ratio  
        New_data_r["Mixt_Init"] = mixture  
    
        all_data_r = pd.concat([all_data_r,New_data_r],ignore_index=True)    
        
    all_data_d.to_csv(os.path.join(Path,f"Processing_{name_d}.csv"))
    all_data_r.to_csv(os.path.join(Path,f"Processing_{name_r}.csv"))

    return all_data_d , all_data_r
    
def Calc_ai_delay(time,temp) :
    loc_time = time
    loc_temp = temp
    diff_temp = np.diff(loc_temp)/np.diff(loc_time)
    ign_pos = np.argmax(diff_temp)
    output=loc_time[ign_pos]
    return output

def shift_1D(grid: list, T: list) -> list:
    gradient = np.gradient(T, grid)
    indice_gradient = np.argmax(gradient)
    shift_grid = grid - grid.loc[indice_gradient]

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

 