import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def Calculate_AED_ML(data_d,data_r,input,Path,flag_output) :
    species = [species for species, values in input.items() if values["AED"] == 1]
    
    data_d_log=data_d[species+["T","IDT"]].apply(np.log)  
    data_r_log=data_r[species+["T","IDT"]].apply(np.log) 
    
    scl = StandardScaler()
    scl.fit(data_d_log)
    data_d_log_scl = pd.DataFrame(scl.transform(data_d_log),columns= data_d_log.columns)
    data_r_log_scl = pd.DataFrame(scl.transform(data_r_log),columns= data_r_log.columns)
        
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d_log_scl[s]-data_r_log_scl[s])
    Err["T"] = np.abs(data_d_log_scl["T"]-data_r_log_scl["T"])
    Err["IDT"] = np.abs(data_d_log_scl["IDT"]-data_r_log_scl["IDT"])
    Err_AED = np.sum(np.sum(Err,axis=0))
    print(f"Err AED ML = {Err_AED:0.2E}")
    
    if flag_output == True : 
        return Err_AED, Err 
    
    else :
        return Err_AED
