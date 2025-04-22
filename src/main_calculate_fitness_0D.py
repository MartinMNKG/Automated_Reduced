from tools import * 
main_path = os.getcwd()
Name_Folder = "0D"
Path = os.path.join(main_path,f"{Name_Folder}")

data_d = pd.read_csv("/work/kotlarcm/WORK/Automated_Reduced/0D/Processing_Detailed.csv")
data_r = pd.read_csv("/work/kotlarcm/WORK/Automated_Reduced/0D/Processing_Reduced.csv")

nb_case_0D = data_d["P_Init"].nunique()*  data_d["T_Init"].nunique()  *  data_d["Phi_Init"].nunique()  *  data_d["Mixt_Init"].nunique() 
lenght= int(data_d.shape[0]/ nb_case_0D)

   
species_AED= [col for col in data_r.columns if col.startswith("Y_")]
# species_AED = ["Y_NH3","Y_H2","Y_H2O","Y_NO"]
Calculate_AED(data_d,data_r,species_AED,Path)
    

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
Err_Orch, Err_Orch_species = Calculate_ORCH(data_d,data_r,Species_ORCH,coefficients_ORCH,eps,Path)
print(f"Fitness Orch = {Err_Orch}")


#PMO : 
Intergrate_Species =["Y_H2", "Y_NH3", "Y_O2", "Y_OH","Y_NO", 'Y_H2O','Y_NO2', 'Y_N2O','Y_N2']
Peak_species = ['Y_H', 'Y_O', 'Y_HO2', 'Y_N', 'Y_N2H2', 'Y_HNO',"Y_NH","Y_NH2","Y_NNH"]
Err_PMO , F1 ,F2, F3 ,F4 =Calculate_PMO(data_d,data_r,Intergrate_Species,Peak_species,nb_case_0D,lenght,Path) 
print(f"Fitness PMO = {Err_PMO:.3e}")

    