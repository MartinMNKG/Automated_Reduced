import pandas as pd 
import os
import re
import itertools


class MinMaxScaler:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)

    def transform(self, x):
        x = (x - self.min) / (self.max - self.min + 1e-7)
        return x

    def inverse_transform(self, x):
        x = self.min + x * (self.max - self.min + 1e-7)
        return x
    
def generate_test_cases_bifuel(pressure_range,temp_range, second_param,mixture):
    test_cases = list(itertools.product(pressure_range, temp_range, second_param,mixture))

    # Convertir les pressions en Pascals (Pa) car Cantera utilise les Pascals
    test_cases = [(p * 101325, T, second,mixture) for p, T, second,mixture in test_cases]
    
    return test_cases

def Create_directory(main_path,name) : 
    
    dossier = os.path.join(main_path,f"{name}")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass
    return dossier


def extract_values(path):
    # Regex améliorée : on impose que P soit suivi de chiffres/point AVANT ".csv"
    match = re.search(r'ER([0-9.]+)_T([0-9.]+)_P([0-9.]+)\.csv$', path)
    if match:
        try:
            er = float(match.group(1))
            temp = float(match.group(2))
            press = float(match.group(3))
            return (er, temp, press)
        except ValueError:
            pass  # en cas de float invalide
    # Cas d'erreur ou fichier mal nommé
    print(f"⚠️ Mauvais format de fichier : {path}")
    return (float('inf'), float('inf'), float('inf'))

def concat_csv_list(csv_paths: list[str]) -> pd.DataFrame:
    """
    Concatène une liste de fichiers CSV en un seul DataFrame pandas.

    Args:
        csv_paths (list[str]): Liste des chemins vers les fichiers CSV.

    Returns:
        pd.DataFrame: DataFrame concaténé contenant les données de tous les fichiers.
    """
    all_data = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            all_data.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {path}: {e}")

    return pd.concat(all_data, ignore_index=True)