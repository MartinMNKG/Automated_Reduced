from tools import *
main_path = os.getcwd()


launch = False 

fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"

Detailed_file = "./data/detailed.yaml"
Reduced_file = "./data/reduced.yaml"

gas_det = ct.Solution(Detailed_file)
gas_red = ct.Solution(Reduced_file)
_gas_det_copy = ct.Solution(Detailed_file)
_gas_red_copy = ct.Solution(Reduced_file)

## 0D reactor : 
if launch == True : 
    
    Name_Folder = "0D"
    Path = Create_0D_directory(main_path,Name_Folder)
    pressure_0D = np.linspace(1,1,1).tolist()
    temperature_0D = np.linspace(1000,2000,5).tolist()
    phi_0D = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
    mixture_0D =np.linspace(0.85,0.85,1).tolist()
    tmax = 0.05
    dt= 1e-6
    case_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)
    Sim0D(gas_det,_gas_det_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,"Detailed",Path)
    Sim0D(gas_red,_gas_red_copy,fuel1,fuel2,oxidizer,case_0D,dt,tmax,"Reduced",Path)
