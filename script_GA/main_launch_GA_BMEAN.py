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

from GA.main_GA import Launch_GA 

#Create Calcul Folder 
Name_Folder = "/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/SELECTED_SPECIES/BROOKESIA_MEAN"


# Fitness used 
Fitness = Calculate_Brookesia_MEAN
input_fitness = ["Y_NH3","Y_NH","Y_NO","Y_NNH","Y_HNO","Y_N2O","Y_H2","Y_H2O","Y_OH","Y_O","Y_H"]
 

# Cantera inputs 
Detailed_file = "/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/data/detailed.yaml"
Reduced_file = "/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/data/reduced.yaml"
fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.78, AR : 0.01"

#0D inputs 
tmax = 0.1
dt= 1e-6
length = 1000
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1300,1300,1).tolist() #1300 
phi_0D = [0.5,1.5,6,13] # Luc Data 
mixture_0D =np.linspace(0.85,0.85,1).tolist()
cases_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)
    
# GA input 
pop_size = 64 # 500 
ngen =500 # 100 
elitism_size = int(pop_size*10/100)
cxpb = 1
mutpb = 0.3

type_fit = "Maxi" #for ORCH, AED, AEDML, PMO  //// Or Maxi for BROOKESIA
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