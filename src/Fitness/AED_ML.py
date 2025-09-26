import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def Calculate_AED_ML(data_d,data_r,input,flag_output) :
    
    eps = 1.0e-12
    if isinstance(input, list) and input:
        species = input
    else:
        species = [col for col in data_r.columns if col.startswith("Y_")]
    data_d[data_d< eps] = eps 
    data_r[data_r< eps] = eps
    
    data_d_log=data_d[species].apply(np.log)  
    data_r_log=data_r[species].apply(np.log) 
    
    data_d_log["T"] = data_d["T"]
    data_d_log["IDT"] = data_d["IDT"]
    
    data_r_log["T"] = data_r["T"]
    data_r_log["IDT"] = data_r["IDT"]
    
    
    scl = StandardScaler()
    scl.fit(data_d_log)
    data_d_log_scl = pd.DataFrame(scl.transform(data_d_log),columns= data_d_log.columns)
    data_r_log_scl = pd.DataFrame(scl.transform(data_r_log),columns= data_r_log.columns)
        
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d_log_scl[s]-data_r_log_scl[s])
    # Err["T"] = np.abs(data_d_log_scl["T"]-data_r_log_scl["T"])
    # Err["IDT"] = np.abs(data_d_log_scl["IDT"]-data_r_log_scl["IDT"])
    Err_AED = np.sum(np.sum(Err,axis=0))
    # print(f"Err AED ML = {Err_AED:0.2E}")
    
    if flag_output == True : 
        return Err_AED, Err 
    
    else :
        return Err_AED
