import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os
   
def Calculate_ORCH(data_d,data_r,species,coefficient,eps,Path): 
    Err_ORCH = np.abs(data_d[species]-data_r[species])/np.maximum(np.abs(data_d[species]),eps)
    mask = np.abs(data_d[species])<eps
    Err_ORCH[mask] = 0
    
    value_fitness_species =[]
    for s in species : 
        if s in coefficient : 
            k = coefficient[s]
        else :
            k = 0.05
        
        value_fitness_species.append(k*np.sum(Err_ORCH[s]))
    
    plt.figure()
    sns.boxplot(data=Err_ORCH,showfliers=False)
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(Path,"ORCH.png"))
    
    return np.sum(value_fitness_species),value_fitness_species