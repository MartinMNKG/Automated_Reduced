from tools import *

main_path = os.getcwd()

launch = True 
save = False 

start_time = time.time()


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
    
    
    data_ref = Sim1D(gas_det,fuel1,fuel2,oxidizer,case_1D,Name_Ref,Path,save)
    
    print(f"Time Simu Ref : { time.time()- start_time}")
    simu_data = time.time()
    
    data = Sim1D(gas_red,fuel1,fuel2,oxidizer,case_1D,Name_Data,Path,save)
    
    print(f'Time Simu Data = { time.time() - simu_data }')
    process = time.time()
    
    Processing_Ref  = Processing_1D_PMX_ref(data_ref,case_1D,Name_Ref,Path,save)
    Processing_Data = Processing_1D_PMX_data(data,Processing_Ref,case_1D,Name_Data,Path,save)
    
    print(f"Time Processing = { time.time() - process }")
    Processing_Ref.to_csv(os.path.join(Path, f"Processing_{Name_Ref}.csv"))
    Processing_Data.to_csv(os.path.join(Path, f"Processing_{Name_Data}.csv"))
    
else  : 
    
    List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
    List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
    Processing_Ref2,Processing_Data2 = Launch_processing_1D_PMX_csv(List_Ref,List_Data,case_1D,Name_Ref,Name_Data,Path,save)
    
print(f"End = { time.time() - start_time }")