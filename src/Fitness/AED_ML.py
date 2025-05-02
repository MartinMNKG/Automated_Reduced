import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def Calculate_AED_ML(data_d,data_r,input,Path,flag_output) :
    species = [species for species, values in input.items() if values["AED"] == 1]
    
    data_d[species]=data_d[species].apply(np.log)  
    data_r[species]=data_r[species].apply(np.log) 
    
    scl = StandardScaler()
    scl.fit(data_d[species+["T"]])
    data_d[species+["T"]] = scl.transform(data_d[species+["T"]])
    data_r[species+["T"]] = scl.transform(data_r[species+["T"]])
        
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d[s]-data_r[s])
    Err["T"] = np.abs(data_d["T"]-data_r["T"])
    Err["IDT"] = np.abs(data_d["IDT"]-data_r["IDT"])
    Err_AED = np.sum(np.sum(Err,axis=0))
    print(f"Err AED ML = {Err_AED:0.2E}")
    
    if flag_output == True : 
        return Err_AED, Err 
    
    else :
        return Err_AED
