import cantera as ct
import pickle
import numpy as np
import glob
import pandas as pd 
import matplotlib
matplotlib.use("Agg")  # Utilise le backend sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns 
from src.Database.Tools_0D import Sim0D, Processing_0D_data
from src.Database.utils import generate_test_cases_bifuel 
from src.Fitness.AED_ML import Calculate_AED_ML


Processing_Ref = pd.read_csv("/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/Test2_GA_POP10/Processing_Detailed.csv")
Processing_Red = pd.read_csv("/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Processing_Reduced.csv")
fuel1 = "NH3"
fuel2 = "H2"
oxidizer = "O2:0.21, N2:0.79, AR : 0.01"
tmax = 0.1
dt= 1e-6
length = 1000
pressure_0D = np.linspace(1,1,1).tolist()
temperature_0D = np.linspace(1300,1300,1).tolist() #1300 
phi_0D = [0.5,1.5,6,13] # Luc Data 
mixture_0D =np.linspace(0.85,0.85,1).tolist()
cases_0D = generate_test_cases_bifuel(pressure_0D,temperature_0D,phi_0D,mixture_0D)

nmech = 500
Path = "/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES"
launch = True
if launch == True : 

    gas_AEDML = ct.Solution(f"./CALCUL/ALL_SPECIES/AED_ML/mech/Mech_gen_{nmech}.yaml")
    gas_BMEAN = ct.Solution(f"./CALCUL/ALL_SPECIES/BROOKESIA_MEAN/mech/Mech_gen_{nmech}.yaml")    
    gas_BMAX = ct.Solution(f"./CALCUL/ALL_SPECIES/BROOKESIA_MAX2/mech/Mech_gen_{nmech}.yaml")
    gas_ORCH = ct.Solution(f"./CALCUL/ORCH/GIOVANNI_SELECTION/mech/Mech_gen_{nmech}.yaml")


    SIM_AEDML = Sim0D(gas_AEDML,gas_AEDML,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
    SIM_BMEAN = Sim0D(gas_BMEAN,gas_BMEAN,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
    SIM_BMAX = Sim0D(gas_BMAX,gas_BMAX,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)
    SIM_ORCH = Sim0D(gas_ORCH,gas_ORCH,fuel1,fuel2,oxidizer,cases_0D,dt,tmax,"","",False)



    PROCESS_AEDML = Processing_0D_data(SIM_AEDML,Processing_Ref,cases_0D,f"AEDML_mech{nmech}",Path,True)
    PROCESS_BMEAN= Processing_0D_data(SIM_BMEAN,Processing_Ref,cases_0D,f"BMEAN_mech{nmech}",Path,True)
    PROCESS_BMAX= Processing_0D_data(SIM_BMAX,Processing_Ref,cases_0D,f"BMAX_mech{nmech}",Path,True)
    PROCESS_ORCH= Processing_0D_data(SIM_ORCH,Processing_Ref,cases_0D,f"ORCH_mech{nmech}",Path,True)
    

PROCESS_AEDML = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Processing_AEDML_mech{nmech}.csv")
PROCESS_BMEAN = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Processing_BMEAN_mech{nmech}.csv")
PROCESS_BMAX = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Processing_BMAX_mech{nmech}.csv")
PROCESS_ORCH = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/Processing_ORCH_mech{nmech}.csv")



Err_Red,_Err_Red = Calculate_AED_ML(Processing_Ref,Processing_Red,[],True)
Err_AEDML,_Err_AEDML = Calculate_AED_ML(Processing_Ref,PROCESS_AEDML,[],True)
Err_BMEAN,_Err_BMEAN = Calculate_AED_ML(Processing_Ref,PROCESS_BMEAN,[],True)
Err_BMAX,_Err_BMAX = Calculate_AED_ML(Processing_Ref,PROCESS_BMAX,[],True)
Err_ORCH,_Err_ORCH = Calculate_AED_ML(Processing_Ref,PROCESS_ORCH,[],True)

# Err_Red,_Err_Red = Calculate_AED(Processing_Ref,Processing_Red,[],True)
# Err_AED,_Err_AED = Calculate_AED(Processing_Ref,PROCESS_AED,[],True)
# Err_AEDML,_Err_AEDML = Calculate_AED(Processing_Ref,PROCESS_AEDML,[],True)
# Err_BMEAN,_Err_BMEAN = Calculate_AED(Processing_Ref,PROCESS_BMEAN,[],True)
# Err_BMAX,_Err_BMAX = Calculate_AED(Processing_Ref,PROCESS_BMAX,[],True)
# Err_ORCH,_Err_ORCH = Calculate_AED(Processing_Ref,PROCESS_ORCH,[],True)

df_reduced = _Err_Red.melt(var_name="Species", value_name="Err")
df_reduced["Scheme"] = "Reduced"

df_optimAEDML = _Err_AEDML.melt(var_name="Species", value_name="Err")
df_optimAEDML["Scheme"] = f"AEDML_mech{nmech}"

df_optimBMEAN = _Err_BMEAN.melt(var_name="Species", value_name="Err")
df_optimBMEAN["Scheme"] = f"BMEAN_mech{nmech}"

df_optimBMAX = _Err_BMAX.melt(var_name="Species", value_name="Err")
df_optimBMAX["Scheme"] = f"BMAX_mech{nmech}"


df_optimORCH = _Err_ORCH.melt(var_name="Species", value_name="Err")
df_optimORCH["Scheme"] = f"ORCH_mech{nmech}"

df_all = pd.concat([df_reduced,df_optimAEDML,df_optimBMEAN,df_optimBMAX,df_optimORCH], ignore_index=True)

palette = {
    "Reduced": "#e41a1c",        # Rouge vif
    # f"AED_mech{nmech}": "#377eb8",   # Bleu moyen
    f"AEDML_mech{nmech}": "#4daf4a", # Vert moyen
    f"BMEAN_mech{nmech}": "#984ea3", # Violet foncé
    f"BMAX_mech{nmech}": "#ff7f00",   # Orange
    f"ORCH_mech{nmech}" : "#a65628" 
}

# Tracer
plt.figure(figsize=(16, 6))
sns.boxplot(data=df_all, x="Species", y="Err", hue="Scheme",  palette=palette,showfliers=False)
plt.yscale("log")
plt.ylim([1e-4, 5e1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ALL_SPECIES/AED_ML_gen{nmech}_SPECIES.png")
