from ORCH import Calculate_ORCH
from PMO import Calculate_PMO
from AED import Calculate_AED 
from Brookesia import Calcualte_Brookesia 
from ruamel.yaml import YAML
import numpy as np
import os 
import pandas as pd 
yaml = YAML()
main_path = os.getcwd()
Name_Folder = "0D"
Path = os.path.join(main_path,f"{Name_Folder}")

with open("./data/Info_species_fitness.yaml", "r") as f:
    data = yaml.load(f)
    
data_d = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_Detailed.csv")
data_r = pd.read_csv(f"/work/kotlarcm/WORK/Automated_Reduced/{Name_Folder}/Processing_Reduced.csv")

# fitness = [Calculate_AED, Calculate_ORCH , Calculate_PMO]
fitness = [Calcualte_Brookesia]

for F in fitness : 
    Err = F(data_d,data_r,data,Path)
    print(Err)
    


    