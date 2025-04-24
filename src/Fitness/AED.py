import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os

def Calculate_AED(data_d,data_r,data,Path) :
    species = list(data.keys()) 
    Err = pd.DataFrame()       
    for s in species : 
        Err[s] = np.abs(data_d[s]-data_r[s])
    Err["T"] = np.abs(data_d["T"]-data_r["T"])
    Err["IDT"] = np.abs(data_d["IDT"]-data_r["IDT"])
    
    plt.figure()
    sns.boxplot(data=Err,showfliers=False)
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(Path,"AED.png"))
    Err_AED = np.sum(np.sum(Err))
    print(f"Err AED = {Err_AED}")
    return Err_AED