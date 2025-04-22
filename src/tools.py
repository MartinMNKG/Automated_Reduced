import numpy as np
import cantera as ct
import pandas as pd 
import os
import sys
import time
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

def Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case_0D,dt,tmax,type,dossier):
    dossier = Create_directory(dossier,type)
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
        species_d = [col for col in data_d.columns if col.startswith("Y_")]
        IDT_d = Calc_ai_delay(data_d["t"],data_d["T"])
        if time_shift == True : 
            data_d["t"] = data_d["t"] - IDT_d
        
        if log == True : 
            data_d[species_d]=data_d[species_d].apply(np.log)  
        
        New_data_d["common_grid"] = Generate_common_grid(data_d["t"],lenght)
        
        for s in species_d : 
            int_func = interp1d(data_d["t"],data_d[s],fill_value='extrapolate')
            New_data_d[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_d["t"],data_d["T"],fill_value='extrapolate')
        New_data_d["T"] = int_func(New_data_d["common_grid"])
        
        all_scl =[]
        if scaler== True : 
            for s in species_d : 
                scl = MinMaxScaler()
                scl.fit(New_data_d[s])
                New_data_d[s] = scl.transform(New_data_d[s])
                all_scl.append(scl)
    
        
        New_data_d["T"]= data_d["T"]
        New_data_d["IDT"] = IDT_d
        New_data_d["P_Init"]  = pressure
        New_data_d["T_Init"]  = temperature
        New_data_d["phi_ini"]  =   equivalence_ratio  
        New_data_d["mixt_init"] = mixture           
        
        all_data_d = pd.concat([all_data_d,New_data_d],ignore_index=True)
    
        
        New_data_r = pd.DataFrame()
        data_r=pd.read_csv(csv_r)
        species_r = [col for col in data_r.columns if col.startswith("Y_")]
        IDT_r = Calc_ai_delay(data_r["t"],data_r["T"])
        if time_shift == True : 
            data_r["t"] = data_r["t"] - IDT_r
        if log == True : 
            data_r[species_r]=data_r[species_r].apply(np.log)  
        
        New_data_r["common_grid"] = New_data_d["common_grid"]
        
        for s in species_r : 
            int_func = interp1d(data_r["t"],data_r[s],fill_value='extrapolate')
            New_data_r[s] = int_func(New_data_d["common_grid"])
            
        int_func = interp1d(data_r["t"],data_r["T"],fill_value='extrapolate')
        New_data_r["T"] = int_func(New_data_d["common_grid"])
            
        if scaler== True : 
            for s in species_r : 
                scl = all_scl[species_d.index(s)]
                New_data_r[s] = scl.transform(New_data_r[s])
        
        
        New_data_r["T"]= data_r["T"]
        New_data_r["IDT"] = IDT_r
        New_data_r["P_Init"]  = pressure
        New_data_r["T_Init"]  = temperature
        New_data_r["phi_ini"]  =   equivalence_ratio  
        New_data_r["mixt_init"] = mixture  
    
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
        
        New_data_d["common_grid"] = Generate_common_grid(data_d["grid"],lenght)
        
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
        New_data_d["phi_ini"]  =   equivalence_ratio  
        New_data_d["mixt_init"] = mixture     
        
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
        New_data_r["phi_ini"]  =   equivalence_ratio  
        New_data_r["mixt_init"] = mixture  
    
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

def Generate_common_grid(time,lenght) :
    return np.linspace(min(time),max(time),lenght)

def Calculate_AED(data_d,data_r,species,Path) : 
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d[s]-data_r[s])
    Err["T"] = np.abs(data_d["T"]-data_r["T"])
    Err["IDT"] = np.abs(data_d["IDT"]-data_r["IDT"])
    plt.figure()
    sns.boxplot(data=Err,showfliers=False)
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(Path,"AED.png"))
    
    
def Calculate_ORCH(data_d,data_r,species,coefficient,eps): 
    Err_ORCH = np.abs(data_d[species]-data_r[species])/np.maximum(np.abs(data_d[species]),eps)
    mask = np.abs(data_d[species])<eps
    Err_ORCH[mask] = 0
    
    value_fitness_species =[]
    for s in species : 
        if s in coefficient : 
            k = coefficient[s]
        else :
            k = 0.05
        
        value_fitness_species.append(k*np.sum(Err_ORCH[s]))
    
    return np.sum(value_fitness_species),value_fitness_species

    

def Calculate_PMO(data_d,data_r,integral,peak,case,lenght) : 
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    for c in range(len(case)) : 
        
        loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
        loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
        
        loc_F1 = []
        for si in integral : 
            top1 = np.trapezoid((np.abs(loc_data_r[si]-loc_data_d[si])),loc_data_d["common_grid"])
            bot1 = np.trapezoid(np.abs(np.array(loc_data_r[si])), np.array(loc_data_d["common_grid"]))
            loc_F1.append((top1 / bot1) ** 2 if bot1 != 0 else 0)
        
        loc_F2 =[] 
        for sp in peak : 
            top2 = np.max(loc_data_d[sp])-np.max(loc_data_r[sp])
            bot2 = np.max(loc_data_d[sp])
            loc_F2.append((top2 / bot2) ** 2 if bot2 != 0 else 0)
        
        top3 = np.trapezoid(np.abs(loc_data_r["T"] - loc_data_d["T"]), loc_data_d["common_grid"])
        bot3 = np.trapezoid(np.abs(loc_data_d["T"]), loc_data_d["common_grid"])
        F3.append((top3 / bot3) ** 2 if bot3 != 0 else 0)
        
        top4 = loc_data_r["IDT"][0] - loc_data_d["IDT"][0]
        bot4 = loc_data_d["IDT"][0]
        F4.append((top4 / bot4) ** 2 if bot4 != 0 else 0)
        
        return F1, F2, F3, F4