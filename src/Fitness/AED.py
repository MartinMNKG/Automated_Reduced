import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os

def Calculate_AED(data_d,data_r,input,flag_output) :
    species = input
    
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d[s]-data_r[s])
    Err["T"] = np.abs(data_d["T"]-data_r["T"])
    Err["IDT"] = np.abs(data_d["IDT"]-data_r["IDT"])
    Err_AED = np.sum(np.sum(Err,axis = 0))
    print(f"Err AED = {Err_AED:0.2E}")
    
    if flag_output == True : 
        return Err_AED, Err 
    
    else :
        return Err_AED
