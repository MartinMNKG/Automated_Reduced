
import numpy as np
import os 
import sys
import pandas as pd 
from ruamel.yaml import YAML
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))




from Fitness.ORCH import Calculate_ORCH
from Fitness.PMO import Calculate_PMO
from Fitness.AED import Calculate_AED 
from Fitness.AED_ML import Calculate_AED_ML
from Fitness.Brookesia import Calculate_Brookesia 

yaml = YAML()
main_path = os.getcwd()
flag_output = False
Name_Folder = "0D"
Name_File = "OptimB"
Path = os.path.join(main_path,f"{Name_Folder}")

with open("./data/Info_species_fitness.yaml", "r") as f:
    Input = yaml.load(f)
    
data_d = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_Detailed.csv")
data_r = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_{Name_File}.csv")

fitness = [Calculate_AED, Calculate_ORCH , Calculate_PMO, Calculate_Brookesia,Calculate_AED_ML]
# fitness = [Calculate_AED_ML] 

for F in fitness : 
    Err = F(data_d,data_r,Input,Path,flag_output)
    


    