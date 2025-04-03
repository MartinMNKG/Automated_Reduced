from tools import *
main_path = os.getcwd()


launch = False 

Processing = False 
Time_shift= False  
log = False 
scaler= False

AED = True 


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
Path = Create_0D_directory(main_path,Name_Folder)
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1000,2000,5).tolist()
phi_0D = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
mixture_0D =np.linspace(0.85,0.85,1).tolist()
tmax = 0.05
dt= 1e-6
case_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)
if launch == True : 
    #Launch 0D reactor Base
    Sim0D(gas_det,_gas_det_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,name_d,Path)
    Sim0D(gas_red,_gas_red_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,name_r,Path)

if Processing == True : 
    #Processing Database
    lenght = 500
    csv_d = glob.glob(os.path.join(Path,f"{name_d}/*.csv"))
    csv_r = glob.glob(os.path.join(Path,f"{name_r}/*.csv"))
    data_d , data_r = Processing_0D(csv_d,csv_r,case_0D,Time_shift,log,scaler,lenght,name_d,name_r,Path) 


if AED == True :  
    if Processing == False :  
        data_d = pd.read_csv(os.path.join(Path,f"Processing_{name_d}.csv"))
        data_r = pd.read_csv(os.path.join(Path,f"Processing_{name_r}.csv"))
    
    species= [col for col in data_r.columns if col.startswith("Y_")]
    species = ["Y_NH3","Y_H2","Y_H2O","Y_NO"]
    Calculate_AED(data_d,data_r,species,Path)