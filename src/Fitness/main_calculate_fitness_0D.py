from ORCH import Calculate_ORCH
from PMO import Calculate_PMO
from AED import Calculate_AED 
from AED_ML import Calculate_AED_ML
from Brookesia import Calculate_Brookesia 
from ruamel.yaml import YAML
import numpy as np
import os 
import pandas as pd 
yaml = YAML()
main_path = os.getcwd()
flag_output = False
Name_Folder = "0D_1case"
Name_File = "Reduced"
Path = os.path.join(main_path,f"{Name_Folder}")

with open("./data/Info_species_fitness.yaml", "r") as f:
    Input = yaml.load(f)
    
data_d = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_Detailed.csv")
data_r = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_{Name_File}.csv")

# fitness = [Calculate_AED, Calculate_ORCH , Calculate_PMO, Calculate_Brookesia]
fitness = [Calculate_AED_ML] 

for F in fitness : 
    Err = F(data_d,data_r,Input,Path,flag_output)
    


    