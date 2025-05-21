import time 
import os 
import glob 
import numpy as np 
import cantera as ct 

from ..Database.utils import generate_test_cases_bifuel, Create_directory
from ..Database.Tools_1DCF import Sim1D_CF_Extinction, Processing_1D_CF_ref, Processing_1D_CF_data, Launch_processing_1D_CF_csv



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

Name_Folder = "1D_CF_Ex"
Path = Create_directory(main_path,Name_Folder)

pressure_1DCF = np.linspace(1,1,1).tolist()
temperature_1DCF = np.linspace(800,900,2).tolist() # Temperature of the Oxydizer 
strain_1DCF = np.round(np.linspace(1.0, 1.0, 1), 1).tolist() # Strain 
mixture_1DCF =np.linspace(0.85,0.85,1).tolist()



case_1DCF = generate_test_cases_bifuel(pressure_1DCF,temperature_1DCF,strain_1DCF,mixture_1DCF)
if launch == True : 
    # Launch 1D database 
    start = time.time()
    data_ref = Sim1D_CF_Extinction(gas_det,fuel1,fuel2,oxidizer,case_1DCF,Name_Ref,Path,save)
    print(f"Time simu Ref = {time.time()-start }")
    start = time.time() 
    data = Sim1D_CF_Extinction(gas_red,fuel1,fuel2,oxidizer,case_1DCF,Name_Data,Path,save)
    print(f"Time simu Data = {time.time() - start}")
    start = time.time()
    Processing_Ref  = Processing_1D_CF_ref(data_ref,case_1DCF,Name_Ref,Path,save)
    print(f"Time Process Ref = {time.time() - start }")
    start = time.time()
    Processing_Data = Processing_1D_CF_data(data,Processing_Ref,case_1DCF,Name_Data,Path,save)
    print(f"Time Process Data = {time.time() -start}")

    
# else  : 
    
#     List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
#     List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
#     Processing_Ref2,Processing_Data2 = Launch_processing_1D_CF_csv(List_Ref,List_Data,case_1DCF,Name_Ref,Name_Data,Path,save)
    