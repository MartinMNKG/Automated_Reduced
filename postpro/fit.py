import pandas as pd 
import numpy as np
import os 
import glob
import matplotlib
matplotlib.use("Agg")  # Backend pour génération sans GUI
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Fitness.AED import Calculate_AED # type 1
from Fitness.AED_ML import Calculate_AED_ML# type 1
from Fitness.PMO import Calculate_PMO# type 1
from Fitness.ORCH import Calculate_ORCH# type 2
from Fitness.EEM import RE,ABS,RMSE,IE,RE_A # type EEM

def Load_last_csv(path) : 
    csv =glob.glob(os.path.join(path,"hist", "*.csv"))
    csv.sort(key=os.path.getmtime)
    
    df = pd.read_csv(csv[-1])
    return df 

REF = pd.read_csv("Test_AEDML/Processing_Detailed.csv")
START = pd.read_csv("Test_AEDML/Processing_Start.csv")

#Add more path if need
Path_AEDML = "./Test_AEDML"
Path_ORCH = "./Test_ORCH"
Path_RMSE = "./Test_RMSE" 
# Add labels
labels = ["AEDML","ORCH","RMSE"]


# Add input fitness for all type : Type 1 (AEDML, PMO), Type 2( Orch), Type 3(EEM)
input_fitness_type1 = ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO','Y_N2']
input_fitness_type2= {
"Y_NO": 6.0,
"Y_NH": 3.5,
"Y_NH2": 3.5,
"Y_NNH": 5.0,
"Y_H2": 3.0,
"Y_NH3": 3.0,
"Y_O2": 3.0,
"Y_OH": 3.0,
"Y_O": 3.0,
"Y_H": 3.0
}
input_fitness_EEM = {
    "species": ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO'],
    "do_log" : True, #True or False
    "norm_type" : None, # "standard" or "minmax"
}

# Add Every calcul for F_ref
AEDML_fit_start = Calculate_AED_ML(REF,START,input_fitness_type1,False)
ORCH_fit_start = Calculate_ORCH(REF,START,input_fitness_type2,False)
RMSE_fit_start = RMSE(REF,START,input_fitness_EEM.get("species"),False,input_fitness_EEM.get("do_log"),input_fitness_EEM.get("norm_type"))
all_fit_start= [AEDML_fit_start,ORCH_fit_start,RMSE_fit_start]


#Add all fit over generation
fit_AEDML = Load_last_csv(Path_AEDML)
fit_ORCH = Load_last_csv(Path_ORCH)
fit_RMSE = Load_last_csv(Path_RMSE)
all_fit = [fit_AEDML,fit_ORCH,fit_RMSE]

# PLOT 
plt.figure() 
i=0
for df,df_ref in zip(all_fit,all_fit_start):
    plt.plot(df["gen"],df["min"]/df_ref,label=labels[i])

    i=i+1
plt.legend()
plt.ylabel(r"$\frac{F_{obj}}{F_{obj}^{ref}}$",size=18)
plt.xlabel("generations")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("fit_FREF.png")
