import time 
import os
import sys 
import glob 
import numpy as np 
import cantera as ct 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Database.utils import generate_test_cases_bifuel, Create_directory
from Database.Tools_1DPMX import Sim1D, Processing_1D_PMX_ref, Processing_1D_PMX_data, Launch_processing_1D_PMX_csv

main_path = os.getcwd()

launch = True 
save = True 

fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"


Detailed_file = "./data/detailed.yaml"
Name_Ref = "Detailed"
Reduced_file = "./data/STEC_B.yaml"
Name_Data = "OptimB"

gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)



################
##   1D PMX   ##
################
Name_Folder = "1D_PMX"
Path = Create_directory(main_path,Name_Folder)

pressure_1D = np.linspace(1,1,1).tolist()
temperature_1D = np.linspace(300,300,1).tolist()
phi_1D = np.round(np.linspace(0.5, 2.0, 16), 1).tolist()
mixture_1D =np.linspace(0.85,0.85,1).tolist()


case_1D = generate_test_cases_bifuel(pressure_1D,temperature_1D,phi_1D,mixture_1D)
    
if launch == True : 
    # Launch 1D database 
    start = time.time()
    data_ref = Sim1D(gas_det,fuel1,fuel2,oxidizer,case_1D,Name_Ref,Path,save)
    print(f"Time simu Ref = {time.time()-start }")
    start = time.time() 
    data = Sim1D(gas_red,fuel1,fuel2,oxidizer,case_1D,Name_Data,Path,save)
    print(f"Time simu Data = {time.time() - start}")
    start = time.time()
    Processing_Ref  = Processing_1D_PMX_ref(data_ref,case_1D,Name_Ref,Path,save)
    print(f"Time Process Ref = {time.time() - start }")
    start = time.time()
    Processing_Data = Processing_1D_PMX_data(data,Processing_Ref,case_1D,Name_Data,Path,save)
    print(f"Time Process Data = {time.time() -start}")
    
else  : 
    
    List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
    List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
    Processing_Ref2,Processing_Data2 = Launch_processing_1D_PMX_csv(List_Ref,List_Data,case_1D,Name_Ref,Name_Data,Path,save)
    