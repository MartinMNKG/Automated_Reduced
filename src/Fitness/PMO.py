import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os



def Calculate_PMO(data_d,data_r,input,flag_output) : 
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)
    species = [s for group in input.values() for s in group]
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    for c in range(case) : 
        
        loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
        loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
        loc_F1 = []
        loc_F2 =[] 
        for s in species : 
            
            if s in input["integrate_species"]: 
                top1 = np.trapezoid((np.abs(loc_data_r[s]-loc_data_d[s])),loc_data_d["common_grid"])
                
                bot1 = np.trapezoid(np.abs(np.array(loc_data_d[s])), np.array(loc_data_d["common_grid"]))
                
                loc_F1.append((top1 / bot1) ** 2 if bot1 != 0 else 0)
            
            
            
            if s in input["peak_species"] :  
                top2 = np.max(loc_data_d[s])-np.max(loc_data_r[s])
                bot2 = np.max(loc_data_d[s])
                loc_F2.append((top2 / bot2) ** 2 if bot2 != 0 else 0)
            
        F1.append(loc_F1)
        F2.append(loc_F2)
        top3 = np.trapezoid(np.abs(loc_data_r["T"] - loc_data_d["T"]), loc_data_d["common_grid"])
        bot3 = np.trapezoid(np.abs(loc_data_d["T"]), loc_data_d["common_grid"])
        F3.append((top3 / bot3) ** 2 if bot3 != 0 else 0)
        
        top4 = loc_data_r["IDT"].iloc[0] - loc_data_d["IDT"].iloc[0]
        bot4 = loc_data_d["IDT"].iloc[0]
        F4.append((top4 / bot4) ** 2 if bot4 != 0 else 0) 
        
        
    Err_PMO = np.sqrt(np.sum(F1)+np.sum(F2)+np.sum(F3)+np.sum(F4))

    print(f"Err PMO = {Err_PMO:0.2e}")
    if flag_output == True :
        return Err_PMO , F1, F2,F3,F4
    else : 
        return Err_PMO 