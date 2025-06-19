import matplotlib
matplotlib.use("Agg")  # Utilise le backend sans interface graphique
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import pandas as pd 

import seaborn as sns

from src.Database.Tools_0D import Sim0D, Processing_0D_data
from src.Database.utils import generate_test_cases_bifuel, Create_directory 
from src.Fitness.AED_ML import Calculate_AED_ML
from src.Fitness.AED import Calculate_AED
from src.Fitness.Brookesia import Calculate_Brookesia_MAX,Calculate_Brookesia_MEAN
from src.Fitness.ORCH import Calculate_ORCH 

Calculs = "ALL_SPECIES"
gen = 500

AED_ML = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/AED_ML/hist/Output_mpi_gen{gen}.csv")
BMEAN = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/BROOKESIA_MEAN/hist/Output_mpi_gen{gen}.csv")
BMAX = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/BROOKESIA_MAX2/hist/Output_mpi_gen{gen}.csv")
ORCH = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/ORCH/GIOVANNI_SELECTION/hist/Output_mpi_gen{gen}.csv")

Processing_Ref = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/Processing_Detailed.csv")
Processing_Data = pd.read_csv(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/Processing_Reduced.csv")
Input = {
    "Y_NO": 6.0,
    "Y_NH": 3.5,
    "Y_NH2": 3.5,
    "Y_NNH": 5.0,
    "Y_H2": 3.0,
    "Y_NH3": 3.0,
    "Y_O2": 3.0,
    "Y_OH": 3.0,
    "Y_O": 3.0,
    "Y_H": 3.0
}
Err_Ref_AEDML = Calculate_AED_ML(Processing_Ref,Processing_Data,[],False)
Err_Ref_BMEAN = Calculate_Brookesia_MEAN(Processing_Ref,Processing_Data,[],False)
Err_Ref_BMAX = Calculate_Brookesia_MAX(Processing_Ref,Processing_Data,[],False)
Err_Ref_ORCH = Calculate_ORCH(Processing_Ref,Processing_Data,Input,False)

# Style général
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

# Palette personnalisée
colors = {
    "AED ML": "#4daf4a",    # Vert
    "AED": "#377eb8",       # Bleu
    "BMEAN": "#984ea3",     # Violet
    "BMAX": "#ff7f00",      # Orange
    "ORCH": "#e41a1c"       # Rouge
}
print(Err_Ref_AEDML)
# Tracés avec couleurs et labels
plt.plot(AED_ML["gen"], AED_ML["min"] / Err_Ref_AEDML, label="AED ML", color=colors["AED ML"], linewidth=2)
plt.plot(BMEAN["gen"], Err_Ref_BMEAN / BMEAN["max"], label="BMEAN", color=colors["BMEAN"], linewidth=2)
plt.plot(BMAX["gen"], Err_Ref_BMAX / BMAX["max"], label="BMAX", color=colors["BMAX"], linewidth=2)
plt.plot(ORCH["gen"], ORCH["min"] / Err_Ref_ORCH, label="ORCH", color=colors["ORCH"], linewidth=2)

# Axes et titre
# plt.yscale("log")
plt.xlabel("Generations")
plt.ylabel("F/Fref or Fref/F")
plt.title("Evolution of F/Fref (Fref/F) with a  population of 64")

# Légende
plt.legend(title="Fitness", loc="center right")

# Marges et layout
plt.tight_layout()

# Sauvegarde
plt.savefig(f"/home/irsrvhome1/R11/kotlarcm/WORK/OPTIM/CALCUL/{Calculs}/Fit.png", dpi=300)

print( 1/(AED_ML["min"].iloc[100] / Err_Ref_AEDML))
print( 1/(ORCH["min"].iloc[100] / Err_Ref_ORCH))
print( 1/(Err_Ref_BMEAN / BMEAN["max"].iloc[100]))
print(1/(Err_Ref_BMAX / BMAX["max"].iloc[100]))



