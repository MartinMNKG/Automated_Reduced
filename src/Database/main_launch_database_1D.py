from tools import *

main_path = os.getcwd()

launch = True 

Processing = True 
Grid_Shift= False  
log = False 
scaler= False



fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"

Detailed_file = "./data/detailed.yaml"
name_d = "Detailed"
Reduced_file = "./data/reduced.yaml"
name_r = "Reduced"
gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)



################
##   1D PMX   ##
################
Name_Folder = "1D"
Path = Create_directory(main_path,Name_Folder)

pressure_1D = np.linspace(1,1,1).tolist()
temperature_1D = np.linspace(300,300,1).tolist()
phi_1D = np.round(np.linspace(1, 1, 1), 1).tolist()
mixture_1D =np.linspace(0.85,0.85,1).tolist()

lenght = 500
case_1D = generate_test_cases_bifuel(pressure_1D,temperature_1D,phi_1D,mixture_1D)
    
if launch == True : 
    # Launch 1D database 
    Sim1D(gas_det,fuel1,fuel2,oxidizer,case_1D,name_d,Path)
    Sim1D(gas_red,fuel1,fuel2,oxidizer,case_1D,name_r,Path)
    
    
if Processing == True : 
    csv_d = glob.glob(os.path.join(Path,f"{name_d}/*.csv"))
    csv_r = glob.glob(os.path.join(Path,f"{name_r}/*.csv"))
    data_d,data_r = Processing_1D(csv_d,csv_r,case_1D,Grid_Shift,log,scaler,lenght,name_d,name_r,Path)