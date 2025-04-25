#Brookesia : 
# https://pubs.acs.org/doi/full/10.1021/acs.jpca.1c02095 
#https://github.com/Brookesia-py/Brookesia/blob/master/brookesia/Class_def.py#L626

# Plusieur types d'erreurs pour 
# Espèces 
# Temperatures
# SL 
# IDT 
# Trois type : All , Mean, Max 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os

def Calcualte_Brookesia(data_d,data_r,data,Path) :
    Err_calculation = "points" # QoI
    Err_type = "mean"# "all" "mean" "max"
    Err_type_fit ="mean" #max
    
    case = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
    lenght= int(data_d.shape[0]/ case)  
    Err = []
    Err_T = err_estim_T(data_d,data_r,case,lenght,Err_calculation,Err_type)
    
    
    Err.append(Err_T)
    if Err_type_fit =="mean" :
        return 1/np.sum(Err)
    elif Err_type_fit =="max" : 
        return 1/np.max(Err)



def err_estim_T(data_d,data_r,case,lenght,Err_calculation,Err_type) : 
    
    
    if Err_calculation == "points" : 
        diff = []
        if Err_type == "all" : 
            
            for c in range(case) : 
                loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
                loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
                diff.append(np.abs(loc_data_d["T"]-loc_data_r["T"])/np.max(loc_data_d["T"]))
                
        elif Err_type == "mean" : 
        
            for c in range(case) : 
                loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
                loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
                diff_mean = np.trapezoid(np.abs(loc_data_d["T"] - loc_data_r["T"]),loc_data_d["common_grid"])
                diff_ref = np.trapezoid(np.abs(loc_data_d["T"]),loc_data_d["common_grid"])
                diff_red = np.trapezoid(np.abs(loc_data_r["T"]),loc_data_d["common_grid"])
                diff.append(diff_mean/max(diff_ref,diff_red))
                
        elif Err_type =="max" : 
            for c in range(case) : 
                loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
                loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
                
                diff_max_top = np.abs(loc_data_d["T"]- loc_data_r["T"]) 
                diff_max_bot = max(np.max(loc_data_d["T"]), np.max(loc_data_r["T"]))
                diff.append(np.max(diff_max_top / diff_max_bot))

        QoI = diff

    elif Err_calculation =="QoI": 
        QoI = []
        for c in range(case) : 
            loc_data_d = data_d.iloc[c*lenght:c*lenght+lenght]
            loc_data_r = data_r.iloc[c*lenght:c*lenght+lenght]
            
            T_var = (np.max(loc_data_d["T"])-np.min(loc_data_d["T"]))/np.min(loc_data_d["T"])
            
            if T_var > 0.05 : 
                if T_var < 0.1 : 
                    QoI_min = np.abs(np.min(loc_data_d["T"])-np.min(loc_data_r["T"]))/np.min(loc_data_d["T"])
                    QoI_max = np.abs(np.max(loc_data_d["T"])-np.max(loc_data_r["T"]))/np.max(loc_data_d["T"])
                    QoI.append([QoI_min,QoI_max])
                else : 
                    QoI.append( qoi_computation(loc_data_d["common_grid"],loc_data_d["T"],loc_data_r["T"]))
            
    print(np.shape(QoI))      
    return QoI 
                
        
    
def searchNearest(data, search_value, start=0, end_ind=-1):
    if end_ind == -1:
        end_ind = len(data)
    vect = abs(data.iloc[start:end_ind].values - search_value)  # Use .iloc for positional access
    index = vect.argmin()  # Get the index of the minimum value
    value = data.iloc[start:end_ind].iloc[index]  # Access the value using .iloc
    return value, start + index

def qoi_computation(pts_scatter, ref_var, red_var, curve_type='upward'):
    if curve_type == "upward":
        Dmin = [np.amin(ref_var), np.amin(red_var)]
        Dmax = [np.amax(ref_var), np.amax(red_var)]

        # QoI: time at 25% of max data
        D25 = Dmin + (np.array(Dmax) - np.array(Dmin)) * 25 / 100
        D25_ref, ind25_ref = searchNearest(ref_var, D25[0], 1)
        D25_red, ind25_red = searchNearest(red_var, D25[1], 1)

        I25p = abs(pts_scatter.iloc[ind25_red] - pts_scatter.iloc[ind25_ref]) / max(pts_scatter.iloc[ind25_ref], pts_scatter.iloc[ind25_red])

        # QoI: time at 75% of max data
        D75 = Dmin + (np.array(Dmax) - np.array(Dmin)) * 75 / 100
        D75_ref, ind75_ref = searchNearest(ref_var, D75[0], ind25_ref)
        D75_red, ind75_red = searchNearest(red_var, D75[1], ind25_red)
        I75p = abs(pts_scatter.iloc[ind75_red] - pts_scatter.iloc[ind75_ref]) / pts_scatter.iloc[ind75_ref]

        # QoI: maximum value comparison
        Ipa = abs(Dmax[1] - Dmax[0]) / max(Dmax[0], Dmax[1])

        # QoI: diff time 90% / 5% (allow to catch the gradient of concentration)
        D90 = Dmin + (np.array(Dmax) - np.array(Dmin)) * 90 / 100
        D05 = Dmin + (np.array(Dmax) - np.array(Dmin)) * 5 / 100
        D90_ref, ind90_ref = searchNearest(ref_var, D90[0], ind75_ref)
        D05_ref, ind05_ref = searchNearest(ref_var, D05[0], 1)
        D90_red, ind90_red = searchNearest(red_var, D90[1], ind75_red)
        D05_red, ind05_red = searchNearest(red_var, D05[1], 1)

        D_ref = pts_scatter.iloc[ind90_ref] - pts_scatter.iloc[ind05_ref]
        D_red = pts_scatter.iloc[ind90_red] - pts_scatter.iloc[ind05_red]

        if D_ref != 0:
            rate_ref = (D90_ref - D05_ref) / (pts_scatter.iloc[ind90_ref] - pts_scatter.iloc[ind05_ref])
            rate_red = (D90_red - D05_red) / (pts_scatter.iloc[ind90_red] - pts_scatter.iloc[ind05_red])
            Ipr = abs(rate_red - rate_ref) / max(rate_red, rate_ref)  # Corrected variable name
        elif D_ref == D_red:
            Ipr = 0
        else:
            Ipr = 1.

        dt90 = pts_scatter.iloc[ind90_ref + 1] - pts_scatter.iloc[ind90_ref - 1]
        dt05 = pts_scatter.iloc[ind05_ref + 1] - pts_scatter.iloc[ind05_ref - 1]

        if dt90 > 0.05 * D_ref and dt05 > 0.05 * D_ref:  # if mesh is not sufficiently resolved
            Ipr = np.mean([I25p, Ipa])

        QoI = [I25p, Ipr, Ipa]
    return QoI

      

        