import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os

def Calculate_AED(data_d,data_r,input,flag_output) :
    
    if isinstance(input, list) and input:
        species = input
    else:
        species = [col for col in data_r.columns if col.startswith("Y_")]
        
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d[s]-data_r[s])
    # Err["T"] = np.abs(data_d["T"]-data_r["T"])
    # Err["IDT"] = np.abs(data_d["IDT"]-data_r["IDT"])
    Err_AED = np.sum(np.sum(Err,axis = 0))
    Err["Phi_Init"] = data_r["Phi_Init"]
    Err["T_Init"] = data_r["T_Init"]

    
    if flag_output == True : 
        return Err_AED, Err 
    
    else :
        return Err_AED
