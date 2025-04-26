import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    """ cette fonction permet de charger un fichier csv et de le convertir en dataframe
    Args:
        filename (str): chemin du fichier csv

    Returns: 
        pd.DataFrame: dataframe contenant les données du fichier csv
    """
    
    df=pd.read_csv(filename)
    
    return df

