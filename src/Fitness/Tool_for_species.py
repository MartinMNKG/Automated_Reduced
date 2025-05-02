from ruamel.yaml import YAML
import pandas as pd 

# Initialisation du parseur YAML
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# Données
data_r = pd.read_csv("/work/kotlarcm/WORK/Automated_Reduced/0D/Processing_Reduced.csv")
especes = [col for col in data_r.columns if col.startswith("Y_")]

coefficient = {
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

integrate_species = [
    "Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O', 'Y_N2'
]

peak_species = [
    'Y_H', 'Y_O', 'Y_HO2', 'Y_N', 'Y_N2H2', 'Y_HNO', "Y_NH", "Y_NH2", "Y_NNH"
]

Brookesia = [
    "Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O'
]


AED = [
    "Y_H2", "Y_NH3", "Y_O2", "Y_OH", "Y_NO", 'Y_H2O', 'Y_NO2', 'Y_N2O'
]


# Construction du dictionnaire
donnees_especes = {}

for espece in especes:
    donnees_especes[espece] = {
        "coefficient": coefficient.get(espece, None),
        "Integrate": 1 if espece in integrate_species else 0,
        "Peak": 1 if espece in peak_species else 0,
        "Brookesia" : 1 if espece in Brookesia else 0, 
        "AED" : 1 if espece in AED else 0 
    }

# Écriture dans le fichier YAML
with open("./data/Info_species_fitness.yaml", "w") as f:
    yaml.dump(donnees_especes, f)
