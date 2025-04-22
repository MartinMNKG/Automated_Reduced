from tools import *
main_path = os.getcwd()


launch = True 

Processing = True 
Time_shift= False  
log = False 
scaler= False

AED = True 
fitness = True 


fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"

Detailed_file = "./data/detailed.yaml"
name_d = "Detailed"
Reduced_file = "./data/reduced.yaml"
name_r = "Reduced"
gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)
_gas_det_copy = ct.Solution(Detailed_file)
_gas_red_copy = ct.Solution(Reduced_file)

################
## 0D reactor ##
################
Name_Folder = "0D"
Path = Create_directory(main_path,Name_Folder)
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1000,2000,5).tolist()
phi_0D = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
mixture_0D =np.linspace(0.85,0.85,1).tolist()

tmax = 0.1
dt= 1e-6
lenght = 500
case_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)
    



if launch == True : 
    #Launch 0D reactor Base
    Sim0D(gas_det,_gas_det_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,name_d,Path)
    Sim0D(gas_red,_gas_red_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,name_r,Path)

if Processing == True : 
    #Processing Database
    csv_d = glob.glob(os.path.join(Path,f"{name_d}/*.csv"))
    csv_r = glob.glob(os.path.join(Path,f"{name_r}/*.csv"))
    data_d , data_r = Processing_0D(csv_d,csv_r,case_0D,Time_shift,log,scaler,lenght,name_d,name_r,Path) 


if AED == True :  
    if Processing == False :  
        data_d = pd.read_csv(os.path.join(Path,f"Processing_{name_d}.csv"))
        data_r = pd.read_csv(os.path.join(Path,f"Processing_{name_r}.csv"))
    
    species_AED= [col for col in data_r.columns if col.startswith("Y_")]
    
    species_AED = ["Y_NH3","Y_H2","Y_H2O","Y_NO"]
    Calculate_AED(data_d,data_r,species_AED,Path)
    

if fitness == True :  
    
    # ORCH 
    Species_ORCH = [col for col in data_r.columns if col.startswith("Y_")]
    eps = 1e-12
    coefficients_ORCH = {
        "Y_NO": 6.0,
        "Y_NH": 3.5,
        "Y_NH2": 3.5,
        "Y_NNH": 5.0,
        "Y_H2": 3.0,
        "Y_NH3": 3.0,
        "Y_O2": 3.0,
        "Y_Y_OH": 3.0,
        "Y_O": 3.0,
        "Y_H": 3.0

    }
    Err_Orch, Err_Orch_species = Calculate_ORCH(data_d,data_r,Species_ORCH,coefficients_ORCH,eps)
    print(f"Fitness Orch = {Err_Orch}")
    
    
    #PMO : 
    Intergrate_Species =["Y_H2", "Y_NH3", "Y_O2", "Y_OH","Y_NO", 'Y_H2O','Y_NO2', 'Y_N2O','Y_N2']
    Peak_species = ['Y_H', 'Y_O', 'Y_HO2', 'Y_N', 'Y_N2H2', 'Y_HNO',"Y_NH","Y_NH2","Y_NNH"]
    F1 ,F2, F3 ,F4 =Calculate_PMO(data_d,data_r,Intergrate_Species,Peak_species,case_0D,lenght) 
    print(f"Fitness PMO = {np.sqrt(np.sum(F1)+np.sum(F2)+np.sum(F3)+np.sum(F4)):.3e}")
    
    