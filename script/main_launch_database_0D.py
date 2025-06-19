import time 
import os
import sys 
import glob 
import numpy as np 
import cantera as ct 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from Database.utils import generate_test_cases_bifuel, Create_directory
from Database.Tools_0D import Sim0D, Processing_0D_ref, Processing_0D_data, Launch_processing_0D_csv


start_simu = time.time()
main_path = os.getcwd()

Name_Folder = "0D_LUC"
launch = True  # Launch Simulation
save = True # Save into CSV 
#################
##   Cantera   ##
#################

fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"

Detailed_file = "./data/detailed.yaml"
Name_Ref = "Detailed"
Reduced_file = "./MECH_ORION/140_ORCH.yaml"
Name_Data = "140_ORCH"

gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)
_gas_det_copy = ct.Solution(Detailed_file)
_gas_red_copy = ct.Solution(Reduced_file)

Path = Create_directory(main_path,Name_Folder)
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1300,1300,1).tolist()
phi_0D = [0.5,1.5,6,13]
mixture_0D =np.linspace(0.85,0.85,1).tolist()

tmax = 0.1
dt= 1e-6

length = 1000
case_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)

if launch == True : 
    #Launch 0D reactor Base
    start = time.time()
    data_ref = Sim0D(gas_det,_gas_det_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,Name_Ref,Path,save) # Return all sim into 1 datafram
    print(f"Time simu Ref = {time.time()-start }")
    start = time.time() 
    data = Sim0D(gas_red,_gas_red_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,Name_Data,Path,save)
    print(f"Time simu Data = {time.time() - start}")
    start = time.time()
    #Process Data Ref and Data 
    Processing_Ref  = Processing_0D_ref(data_ref,case_0D,length,Name_Ref,Path,save) # Return all sim process into 1 datafram 
    print(f"Time Process Ref = {time.time() - start }")
    start = time.time()
    Processing_Data = Processing_0D_data(data,Processing_Ref,case_0D,Name_Data,Path,save)
    print(f"Time Process Data = {time.time() -start}")

else : 

    List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
    List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
    Processing_Ref2,Processing_Data2 = Launch_processing_0D_csv(List_Ref,List_Data,case_0D,length,Name_Ref,Name_Data,Path,save)