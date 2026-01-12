import os
import sys 
import numpy as np 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Database.utils import generate_test_cases_bifuel
from Fitness.AED import Calculate_AED
from Fitness.AED_ML import Calculate_AED_ML
from Fitness.Brookesia import Calculate_Brookesia_MEAN, Calculate_Brookesia_MAX
from Fitness.PMO import Calculate_PMO
from Fitness.ORCH import Calculate_ORCH
from Fitness.EEM import RE,ABS,RMSE,IE,RE_A

from GA.main_GA import Launch_GA 

#Create Calcul Folder 
Name_Folder = "/work/kotlarcm/WORK/Automated_Reduced/Test_RMSE"

# Fitness used 
Fitness = RMSE # Choose fintess : 

### Fitnes type AED Brookesia , PMO : 
# input_fitness = ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO']

### Fitnes type ORCh
# input_fitness = {
# "Y_NO": 6.0,
# "Y_NH": 3.5,
# "Y_NH2": 3.5,
# "Y_NNH": 5.0,
# "Y_H2": 3.0,
# "Y_NH3": 3.0,
# "Y_O2": 3.0,
# "Y_OH": 3.0,
# "Y_O": 3.0,
# "Y_H": 3.0
# }

### Fitness Type EEM (RE,ABS ...)
input_fitness = {
    "species": ['Y_NH3', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_NO', 'Y_NO2', 'Y_N2O',  'Y_NNH', 'Y_HNO',"Y_N2"],
    "do_log" : True, #True or False
    "norm_type" : None, # "standard" or "minmax"
}

# Cantera inputs 
Detailed_file = "/work/kotlarcm/WORK/Automated_Reduced/data/detailed.yaml"
Reduced_file = "/work/kotlarcm/WORK/Automated_Reduced/data/reduced.yaml"
fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.78, AR : 0.01"

#0D inputs 
tmax = 0.1
dt= 1e-6
length = 1000
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1000,1000,1).tolist() #1300 
phi_0D = [1] # Luc Data  # Luc Data 
mixture_0D =np.linspace(0.85,0.85,1).tolist()
cases_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)
    
# GA input 
pop_size = 10 # 500 
ngen =10 # 100 
elitism_size = 1 #int(pop_size*10/100)
cxpb = 1
mutpb = 0.5

type_fit = "Mini" #for ORCH, AED, AEDML, PMO  //// Or Maxi for BROOKESIA
Restart = False 

    
Launch_GA(
    Name_Folder,
    Fitness,
    input_fitness,
    Detailed_file,
    Reduced_file,
    fuel1,
    fuel2,
    oxidizer, 
    tmax, 
    dt, 
    length,
    cases_0D, 
    pop_size, 
    ngen, 
    elitism_size, 
    cxpb,
    mutpb,
    type_fit,
    Restart
    ) 