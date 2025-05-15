from tools import *

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

Name_Folder = "1D_CF"
Path = Create_directory(main_path,Name_Folder)

pressure_1DCF = np.linspace(1,1,1).tolist()
temperature_1DCF = np.linspace(800,800,1).tolist()
strain_1DCF = np.round(np.linspace(500.0, 500.0, 1), 1).tolist()
mixture_1DCF =np.linspace(0.85,0.85,1).tolist()


case_1DCF = generate_test_cases_bifuel(pressure_1DCF,temperature_1DCF,strain_1DCF,mixture_1DCF)
if launch == True : 
    # Launch 1D database 
    data_ref = Sim1D_CF(gas_det,fuel1,fuel2,oxidizer,case_1DCF,Name_Ref,Path,save)
    data = Sim1D_CF(gas_red,fuel1,fuel2,oxidizer,case_1DCF,Name_Data,Path,save)
    
    Processing_Ref  = Processing_1D_CF_ref(data_ref,case_1DCF,Name_Ref,Path,save)
    Processing_Data = Processing_1D_CF_data(data,Processing_Ref,case_1DCF,Name_Data,Path,save)
    
else  : 
    
    List_Ref = glob.glob(os.path.join(Path,f"{Name_Ref}/*.csv"))
    List_Data = glob.glob(os.path.join(Path,f"{Name_Data}/*.csv"))
    Processing_Ref2,Processing_Data2 = Launch_processing_1D_CF_csv(List_Ref,List_Data,case_1DCF,Name_Ref,Name_Data,Path,save)
    