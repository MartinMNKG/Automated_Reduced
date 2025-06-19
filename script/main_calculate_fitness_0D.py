
import numpy as np
import os 
import sys
import pandas as pd 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Fitness.ORCH import Calculate_ORCH
from Fitness.PMO import Calculate_PMO
from Fitness.AED import Calculate_AED 
from Fitness.AED_ML import Calculate_AED_ML
from Fitness.Brookesia import Calculate_Brookesia_MEAN, Calculate_Brookesia_MAX 

main_path = os.getcwd()

flag_output = False
Name_Folder = "0D_LUC" 
Name_Ref = "Detailed"
Name_File = "Gen2" # 
Path = os.path.join(main_path,f"{Name_Folder}")

## Input for each Fitness function, to keep in this order 
Input = [] # If empty, use all species of Reduced 

# Input ={
#     "Y_NO": 6.0,
#     "Y_NH": 3.5,
#     "Y_NH2": 3.5,
#     "Y_NNH": 5.0,
#     "Y_H2": 3.0,
#     "Y_NH3": 3.0,
#     "Y_O2": 3.0,
#     "Y_OH": 3.0,
#     "Y_O": 3.0,
#     "Y_H": 3.0
# }

# Input_PMO = {
#     "integrate_species" : ["Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O', 'Y_N2'] , 
#     "peak_species" : ['Y_H', 'Y_O', 'Y_HO2', 'Y_N', 'Y_N2H2', 'Y_HNO', "Y_NH", "Y_NH2", "Y_NNH"]
# }



data_d = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/{Name_Folder}/Processing_{Name_Ref}.csv")
data_r = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/{Name_Folder}/Processing_{Name_File}.csv")

fitness = [Calculate_AED_ML]

for F in fitness : 
    Err = F(data_d,data_r,Input,flag_output)
    print(F)
    print(Err)
    


    