import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os



def Calculate_PMO(data_d,data_r,data,Path) : 
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)
    species = list(data.keys())
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
            
            if data[s]["Integrate"] == 1 : 
                top1 = np.trapezoid((np.abs(loc_data_r[s]-loc_data_d[s])),loc_data_d["common_grid"])
                bot1 = np.trapezoid(np.abs(np.array(loc_data_r[s])), np.array(loc_data_d["common_grid"]))
                loc_F1.append((top1 / bot1) ** 2 if bot1 != 0 else 0)
            F1.append(loc_F1)
            
            
            if data[s]["Peak"] == 1 :  
                top2 = np.max(loc_data_d[s])-np.max(loc_data_r[s])
                bot2 = np.max(loc_data_d[s])
                loc_F2.append((top2 / bot2) ** 2 if bot2 != 0 else 0)
            F2.append(loc_F2)
        
        top3 = np.trapezoid(np.abs(loc_data_r["T"] - loc_data_d["T"]), loc_data_d["common_grid"])
        bot3 = np.trapezoid(np.abs(loc_data_d["T"]), loc_data_d["common_grid"])
        F3.append((top3 / bot3) ** 2 if bot3 != 0 else 0)
        
        top4 = loc_data_r["IDT"].iloc[0] - loc_data_d["IDT"].iloc[0]
        bot4 = loc_data_d["IDT"].iloc[0]
        F4.append((top4 / bot4) ** 2 if bot4 != 0 else 0) 
        
        
    Err_PMO = np.sqrt(np.sum(F1)+np.sum(F2)+np.sum(F3)+np.sum(F4))

    # Box plot
   
    integrate_species = [species for species, values in data.items() if values["Integrate"] == 1]
    peak_species = [species for species, values in data.items() if values["Peak"] == 1]

    plt.rcParams.update({'font.size': 18})

    # F1
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=np.array(F1), orient="v",showfliers=False)
    plt.xticks(ticks=range(len(integrate_species)), labels=integrate_species, rotation=45)
    plt.ylabel(r'$F_1(Y_i)$')
    plt.yscale("log")
    plt.ylim([1e-4,1e0])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path, "F1.png"))

    # F2
    plt.figure(figsize=(12, 6))
    peak_species = [species for species, values in data.items() if values["Peak"] == 1]
    sns.boxplot(data=np.array(F2), orient="v",showfliers=False)
    plt.xticks(ticks=range(len(peak_species)), labels=peak_species, rotation=45)
    plt.ylabel(r'$F_2(Y_i)$')
    plt.ylim([1e-4,1e1])
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path, "F2.png"))

    # F3
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=np.array(F3), orient="v",showfliers=False)
    plt.xticks(ticks=[0], labels=["T"], rotation=45)
    plt.ylabel(r'$F_3(T)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path, "F3.png"))

    # F4
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=np.array(F4), orient="v",showfliers=False)
    plt.xticks(ticks=[0], labels=["IDT"], rotation=45)
    plt.ylabel(r'$F_4(T)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path, "F4.png"))

    
    print(f"Err PMO = {Err_PMO}")
    
    return Err_PMO
