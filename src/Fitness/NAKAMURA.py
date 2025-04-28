#https://www.sciencedirect.com/science/article/pii/S2666352X23000766#sec0002
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os


def Calculate_Naka(data_d,data_r,data,Path) : 
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)
    species = list(data.keys())
    
    F_loc = []
    for c in range(case) : 
        
        loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
        loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
        
        top = loc_data_r["IDT"].iloc[0]
        bot = loc_data_d["IDT"].iloc[0]
        
        F_loc.append(np.log10(top/bot)**2 if bot != 0 else 0)
        
    Err = np.sqrt(1/case * np.sum(F_loc))
    print(f"Err Naka = {Err}")
    return Err
        