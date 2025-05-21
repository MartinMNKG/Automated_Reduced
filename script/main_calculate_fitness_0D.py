
import numpy as np
import os 
import sys
import pandas as pd 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Fitness.ORCH import Calculate_ORCH
from Fitness.PMO import Calculate_PMO
from Fitness.AED import Calculate_AED 
from Fitness.AED_ML import Calculate_AED_ML
from Fitness.Brookesia import Calculate_Brookesia 

main_path = os.getcwd()

flag_output = False
Name_Folder = "0D" 
Name_Ref = "Detailed"
Name_File = "OptimB" # 
Path = os.path.join(main_path,f"{Name_Folder}")

## Input for each Fitness function, to keep in this order 
Input_AED = [
    "Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O'
]
Input_ORCH ={
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

Input_PMO = {
    "integrate_species" : ["Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O', 'Y_N2'] , 
    "peak_species" : ['Y_H', 'Y_O', 'Y_HO2', 'Y_N', 'Y_N2H2', 'Y_HNO', "Y_NH", "Y_NH2", "Y_NNH"]
}


Input_Brookesia = [
    "Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O'
]


data_d = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_{Name_Ref}.csv")
data_r = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_{Name_File}.csv")

fitness = [Calculate_AED,Calculate_AED_ML, Calculate_ORCH , Calculate_PMO, Calculate_Brookesia]
all_input = [Input_AED, Input_AED,Input_ORCH, Input_PMO,Input_Brookesia]

for F in fitness : 
    Err = F(data_d,data_r,all_input[fitness.index(F)],flag_output)
    


    