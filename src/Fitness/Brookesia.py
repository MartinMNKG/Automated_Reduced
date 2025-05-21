# https://hal.science/hal-03271310/document

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os


def Calculate_Brookesia(data_d,data_r,input,flag_output) : 
    Error_type = "max"
    
    
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)
    species = input
    
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
            top_s = np.abs(np.trapezoid(loc_data_d[s],loc_data_d["common_grid"]) - np.trapezoid(loc_data_r[s],loc_data_r["common_grid"]))
            
            bot_s = np.trapezoid(loc_data_d[s],loc_data_d["common_grid"])
            Err_loc_s.append(top_s/bot_s)
            Err.append(top_s/bot_s)
        Err_s.append(Err_loc_s)
     
    print(f"Err BROOKESIA mean ={1/np.mean(Err):0.2E}")
    print(f"Err BROOKESIA max ={1/np.max(Err):0.2E}") 
       
    if flag_output == True : 
        return Err_s, Err_T, Err_IDT 
    
    else : 
        if Error_type =="mean": 
            
            return 1/np.mean(Err) 
        elif Error_type=="max" : 
             
            return 1/np.max(Err)
    
    
    
    