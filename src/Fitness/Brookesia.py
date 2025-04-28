# https://hal.science/hal-03271310/document

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os


def Calcualte_Brookesia(data_d,data_r,data,Path) : 
    Error_type = "max"
    
    
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)
    species = list(data.keys())
    
    Err_IDT =[] 
    Err_T = []
    Err_s = []
    Err= []
    for c in range(case): 
        loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
        loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
         
        Err_IDT.append(1 - (loc_data_r["IDT"].iloc[0]/loc_data_d["IDT"].iloc[0]))
        Err.append(1 - (loc_data_r["IDT"].iloc[0]/loc_data_d["IDT"].iloc[0]))
        
        top_t = np.abs(np.trapezoid(loc_data_d["T"],loc_data_d["common_grid"]) - np.trapezoid(loc_data_r["T"],loc_data_r["common_grid"]))
        bot_t = np.trapezoid(loc_data_d["T"],loc_data_d["common_grid"])
        Err_T.append(top_t/bot_t)
        Err.append(top_t/bot_t)
        
        
        Err_loc_s =[]
        for s in species : 
            if data[s]["Brookesia"] == 1 : 
                top_s = np.abs(np.trapezoid(loc_data_d[s],loc_data_d["common_grid"]) - np.trapezoid(loc_data_r[s],loc_data_r["common_grid"]))
                bot_s = np.trapezoid(loc_data_d[s],loc_data_d["common_grid"])
                Err_loc_s.append(top_s/bot_s)
                Err.append(top_s/bot_s)
        Err_s.append(Err_loc_s)
        
    plt.figure()
    plt.plot(range(case),Err_IDT)
    plt.xlabel("cases")
    plt.ylabel(r'$Err(IDT)$')
    plt.grid()
    plt.savefig(os.path.join(Path,"Brookesia_Err_IDT.png"))
    
    plt.figure()
    plt.plot(range(case),Err_T)
    plt.xlabel("cases")
    plt.ylabel(r'$Err(T)$')
    plt.grid()
    plt.savefig(os.path.join(Path,"Brookesia_Err_T.png"))
    
    
    plt.figure()
    Brookesia_species = [species for species, values in data.items() if values["Brookesia"] == 1]
    df = pd.DataFrame(Err_s,columns = Brookesia_species)
    sns.boxplot(df, showfliers=False)
    plt.ylabel(r'$Err(Y_i)$')
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path,"Brookesia_Err_S.png"))
    

    
    if Error_type =="mean": 
        return 1/np.mean(Err) 
    elif Error_type=="max" :   
        return 1/np.max(Err)
  
    
    
    