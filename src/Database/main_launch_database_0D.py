from tools import *

start_simu = time.time()
main_path = os.getcwd()

Name_Folder = "0D_Test"
launch = True  # Launch Simulation
save = False # Save into CSV 
#################
##   Cantera   ##
#################

fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"

Detailed_file = "./data/detailed.yaml"
Name_Ref = "Detailed"
Reduced_file = "./data/STEC_B.yaml"
Name_Data = "OptimB"

gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)
_gas_det_copy = ct.Solution(Detailed_file)
_gas_red_copy = ct.Solution(Reduced_file)

Path = Create_directory(main_path,Name_Folder)
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1000,2000,5).tolist()
phi_0D = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
mixture_0D =np.linspace(0.85,0.85,1).tolist()

tmax = 0.1
dt= 1e-6
length = 1000
case_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)

if launch == True : 
    #Launch 0D reactor Base
    
    data_ref = Sim0D(gas_det,_gas_det_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,Name_Ref,Path,save)
    data = Sim0D(gas_red,_gas_red_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,Name_Data,Path,save)
    
    #Process Data Ref and Data 
    Processing_Ref  = Processing_0D_ref(data_ref,case_0D,length,Name_Ref,Path,save)
    Processing_Data = Processing_0D_data(data,Processing_Ref,case_0D,Name_Data,Path,save)
    
    Processing_Ref.to_csv(os.path.join(Path, f"Processing_NoWriting_{Name_Ref}.csv"))
    Processing_Data.to_csv(os.path.join(Path, f"Processing_NoWriting_{Name_Data}.csv"))
    

else : 

    List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
    List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
    Processing_Ref2,Processing_Data2 = Launch_processing_0D_csv(List_Ref,List_Data,case_0D,length,Name_Ref,Name_Data,Path,save)