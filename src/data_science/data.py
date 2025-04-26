import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def rename_columns(data):
    """Cette fonction renomme les colonnes du dataset en français pour une meilleure compréhension
    
    Parameters:
    data : DataFrame, le dataset à traiter

    Returns:
    data : DataFrame, le dataset avec les colonnes renommées

    """
    data.rename(columns={
        'price': 'prix',
        'AreA': 'superficie',
        'bedrooms': 'chambres',
        'BATHROOMS': 'salles_de_bain',
        'stories': 'etages',
        'mainroad': 'proximite_route_principale',
        'guestroom': 'chambre_amis',
        'basement': 'sous_sol',
        'hotwaterheating': 'chauffage_eau_chaude',
        'air conditioning': 'climatisation',
        'parking': 'places_parking',
        'prefarea': 'zone_preferee',
        'furnishing STATUS': 'statut_meublement',
        'houSeaGe': 'age_maison'
    },inplace=True)
    return data


def split_data(df: pd.DataFrame,
               test_ratio: float,
               seed: int) -> tuple[pd.DataFrame]:
    """Diviser le dataset en set de train et test en mélangeant l'ordre des
    données aléatoirement en fixant la random seed.

    Args:
        df (pd.DataFrame): _description_
        test_ratio (float): _description_
        seed (int): _description_

    Returns:
        tuple[pd.DataFrame]: X_train, y_train, X_test, y_test
    """
    # Diviser les données en ensembles d'entraînement et de test sans stratification
    train_data, test_data = train_test_split(df, test_size=test_ratio, random_state=seed)

    # Séparer les variables explicatives (X) et la variable cible (y)
    X_train = train_data.drop(columns=['prix'])
    y_train = train_data['prix']
    X_test = test_data.drop(columns=['prix'])
    y_test = test_data['prix']

    return X_train, X_test, y_train, y_test

def impute_mode_function(data, columns):
    """ cette fonction permet de remplacer les valeurs manquantes par le mode de chaque colonne
    Args:
        data (pd.DataFrame): dataframe contenant les données
        columns (list): liste des colonnes à traiter

    Returns:
        pd.DataFrame: dataframe contenant les données avec les valeurs manquantes remplacées
    
    """
    for column in columns:
        data[column] = data[column].fillna(data[column].mode()[0])



def convert_to_int(data, columns):
    """ cette fonction permet de convertir les colonnes en catégorie
    Args:
        data (pd.DataFrame): dataframe contenant les données
        columns (list): liste des colonnes à traiter

    Returns:
        pd.DataFrame: dataframe contenant les données avec les colonnes converties en catégorie   
    """

    for var in columns:
        data[var] = data[var].astype('int')



def remove_outliers(data, col, threshold=0.01):
    """Retirer les outliers d'une colonne d'un dataset en se basant sur un seuil.
    
    Parameters
    ----------
    data : DataFrame
        Le dataset à traiter.
    col : str
        Le nom de la colonne contenant les outliers.
    threshold : float
        Le seuil à partir duquel une valeur est considérée comme un outlier.

    Returns
    -------
    DataFrame
        Le dataset sans les outliers.
    """
    q1 = data[col].quantile(threshold)
    q3 = data[col].quantile(1-threshold)
    data = data[(data[col] >= q1) & (data[col] <= q3)]
    return data


def encode_boolean_columns(data, columns):
    """Cette fonction permet de convertir les colonnes booléennes en entiers (0 ou 1)
            """
    for col in columns:
        if col in data.columns:
            data[col] = data[col].map({'no': 0, 'yes': 1}).astype(int)
    return data


def encode_statut_meublement(data):
    """Cette fonction permet de convertir la variable 'statut_meublement' en entier

    Parameters:
    data : DataFrame, le dataset à traiter

    Returns:
    data : DataFrame, le dataset avec la variable 'statut_meublement' convertie en entier
    """

    data['statut_meublement'] = data['statut_meublement'].map({
        'unfurnished': 0,
        'furnished': 2,
        'semi-furnished': 1
    }).astype(int)
    return data
