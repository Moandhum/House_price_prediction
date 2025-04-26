import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA


def plot_distributions(data):
    """Cette fonction permet de visualiser les distributions des variables prix et superficie
    
    Parameters:
    data : DataFrame, le dataset à traiter

    Returns:
    None
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.8)

    # Distribution de la variable prix
    sns.histplot(data['prix'], kde=True, color='salmon', ax=ax[0, 0])
    ax[0, 0].set_title('Distribution de la variable prix')
    ax[0, 0].set_ylabel('Fréquence')

    ax[0, 1].boxplot(data['prix'], vert=False)
    ax[0, 1].set_xlabel('Prix')
    ax[0, 1].set_title('Boîte à moustache de la variable prix')

    # Distribution de la variable superficie
    sns.histplot(data['superficie'], kde=True, ax=ax[1, 0])
    ax[1, 0].set_title('Distribution de la variable superficie')
    ax[1, 0].set_ylabel('Fréquence')

    ax[1, 1].boxplot(data['superficie'], vert=False)
    ax[1, 1].set_xlabel('Superficie')
    ax[1, 1].set_title('Boîte à moustache de la variable superficie')

    plt.show()



def plot_categorical_distributions(data):
    """Cette fonction permet de visualiser les distributions des variables catégorielles

    Parameters:
    data : DataFrame, le dataset à traiter

    Returns:
    None
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.8, wspace=0.4)  # Ajustement de l'espace entre les sous-graphiques

    # Distribution de la variable 'chambres'
    sns.countplot(data=data, x='chambres', hue='chambres', palette="viridis", ax=axes[0, 0]) 
    axes[0, 0].set_title("Figure 1: Distribution de la variable 'chambres'")
    axes[0, 0].set_xlabel("Nombre de chambres")
    axes[0, 0].set_ylabel("Fréquence")

    # Distribution de la variable 'salles_de_bain'
    sns.countplot(data=data, x='salles_de_bain', hue='salles_de_bain', palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("Figure 2: Distribution de la variable 'salles de bain'")
    axes[0, 1].set_xlabel("Nombre de salles de bain")
    axes[0, 1].set_ylabel("Fréquence")

    # Distribution de la variable 'etages'
    sns.countplot(data=data, x='etages', hue='etages', palette="viridis", ax=axes[1, 0])
    axes[1, 0].set_title("Figure 3: Distribution de la variable 'étages'")
    axes[1, 0].set_xlabel("Nombre d'étages")
    axes[1, 0].set_ylabel("Fréquence")

    # Distribution de la variable 'places_parking'
    sns.countplot(data=data, x='places_parking', hue='places_parking', palette="viridis", ax=axes[1, 1])
    axes[1, 1].set_title("Figure 4: Distribution de la variable 'places de parking'")
    axes[1, 1].set_xlabel("Nombre de places de parking")
    axes[1, 1].set_ylabel("Fréquence")

    # Distribution de la variable 'statut_meublement'
    sns.countplot(data=data, x='statut_meublement', hue='statut_meublement', palette="viridis", ax=axes[2, 0])
    axes[2, 0].set_title("Figure 5: Distribution de la variable 'statut de meublement'")
    axes[2, 0].set_xlabel("Statut de meublement")
    axes[2, 0].set_ylabel("Fréquence")

    # Distribution de la variable 'proximite_route_principale'
    sns.countplot(data=data, x='proximite_route_principale', hue='proximite_route_principale', palette="viridis", ax=axes[2, 1])
    axes[2, 1].set_title("Figure 6: Distribution de la variable 'proximité de la route principale'")
    axes[2, 1].set_xlabel("Proximité de la route principale")
    axes[2, 1].set_ylabel("Fréquence")

    # Affichage de la figure complète
    plt.show()

def plot_price_relationships(data):
    """Cette fonction permet de visualiser les relations entre le prix et les variables explicatives
    
    Parameters:
    data : DataFrame, le dataset à traiter

    returns:
    None
    """
    fig, axes = plt.subplots(6, 2, figsize=(15, 30))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)  # Ajustement des espacements entre les sous-graphiques

    # Prix en fonction de la superficie
    sns.scatterplot(x='superficie', y='prix', data=data, ax=axes[0, 0])
    axes[0, 0].set_title("Figure 1: Prix en fonction de la superficie")
    axes[0, 0].set_xlabel("Superficie")
    axes[0, 0].set_ylabel("Prix")

    # Prix en fonction du nombre de chambres
    sns.boxplot(x='chambres', y='prix', data=data, ax=axes[0, 1])
    axes[0, 1].set_title("Figure 2: Prix en fonction du nombre de chambres")
    axes[0, 1].set_xlabel("Nombre de chambres")
    axes[0, 1].set_ylabel("Prix")

    # Prix en fonction du nombre de salles de bain
    sns.boxplot(x='salles_de_bain', y='prix', data=data, ax=axes[1, 0])
    axes[1, 0].set_title("Figure 3: Prix en fonction du nombre de salles de bain")
    axes[1, 0].set_xlabel("Nombre de salles de bain")
    axes[1, 0].set_ylabel("Prix")

    # Prix en fonction du nombre d'étages
    sns.boxplot(x='etages', y='prix', data=data, ax=axes[1, 1])
    axes[1, 1].set_title("Figure 4: Prix en fonction du nombre d'étages")
    axes[1, 1].set_xlabel("Nombre d'étages")
    axes[1, 1].set_ylabel("Prix")

    # Prix en fonction du nombre de places de parking
    sns.boxplot(x='places_parking', y='prix', data=data, ax=axes[2, 0])
    axes[2, 0].set_title("Figure 5: Prix en fonction du nombre de places de parking")
    axes[2, 0].set_xlabel("Nombre de places de parking")
    axes[2, 0].set_ylabel("Prix")

    # Prix en fonction du statut de meublement
    sns.boxplot(x='statut_meublement', y='prix', data=data, ax=axes[2, 1])
    axes[2, 1].set_title("Figure 6: Prix en fonction du statut de meublement")
    axes[2, 1].set_xlabel("Statut de meublement")
    axes[2, 1].set_ylabel("Prix")

    # Prix en fonction de la proximité de la route principale
    sns.boxplot(x='proximite_route_principale', y='prix', data=data, ax=axes[3, 0])
    axes[3, 0].set_title("Figure 7: Prix en fonction de la proximité de la route principale")
    axes[3, 0].set_xlabel("Proximité de la route principale")
    axes[3, 0].set_ylabel("Prix")

    # Prix en fonction de la chambre d'amis
    sns.boxplot(x='chambre_amis', y='prix', data=data, ax=axes[3, 1])
    axes[3, 1].set_title("Figure 8: Prix en fonction de la chambre d'amis")
    axes[3, 1].set_xlabel("Chambre d'amis")
    axes[3, 1].set_ylabel("Prix")

    # Prix en fonction du sous-sol
    sns.boxplot(x='sous_sol', y='prix', data=data, ax=axes[4, 0])
    axes[4, 0].set_title("Figure 9: Prix en fonction du sous-sol")
    axes[4, 0].set_xlabel("Sous-sol")
    axes[4, 0].set_ylabel("Prix")

    # Prix en fonction du chauffage à eau chaude
    sns.boxplot(x='chauffage_eau_chaude', y='prix', data=data, ax=axes[4, 1])
    axes[4, 1].set_title("Figure 10: Prix en fonction du chauffage à eau chaude")
    axes[4, 1].set_xlabel("Chauffage à eau chaude")
    axes[4, 1].set_ylabel("Prix")

    # Prix en fonction de la climatisation
    sns.boxplot(x='climatisation', y='prix', data=data, ax=axes[5, 0])
    axes[5, 0].set_title("Figure 11: Prix en fonction de la climatisation")
    axes[5, 0].set_xlabel("Climatisation")
    axes[5, 0].set_ylabel("Prix")

    # Prix en fonction de la zone préférée
    sns.boxplot(x='zone_preferee', y='prix', data=data, ax=axes[5, 1])
    axes[5, 1].set_title("Figure 12: Prix en fonction de la zone préférée")
    axes[5, 1].set_xlabel("Zone préférée")
    axes[5, 1].set_ylabel("Prix")

    # Affichage de la figure complète
    plt.show()



def plot_mca(data_cat):
    """ Cette fonction permet de réaliser une Analyse en Composantes Multiples (ACM) sur les données catégorielles
        et d'afficher le graphique des modalités des variables.
        
    Parameters
    ----------
    data_cat : DataFrame
        Les données catégorielles à traiter.
        
    Returns:
    -------
    None
    """
    # Initialiser le modèle MCA avec 2 composantes principales
    mca = MCA(n_components=2)

    # Ajuster le modèle sur les données catégorielles
    mca.fit(data_cat)

    # Obtenir les coordonnées des modalités des variables
    column_coords = mca.column_coordinates(data_cat)

    # Créer le graphique de l'ACM pour les modalités
    plt.figure(figsize=(8, 6))
    plt.scatter(column_coords[0], column_coords[1], alpha=0.7, c='red')

    # Ajouter des labels aux points pour les modalités
    for i, (x, y) in enumerate(zip(column_coords[0], column_coords[1])):
        plt.text(x, y, column_coords.index[i], ha='right', va='bottom', fontsize=8)

    # Ajouter des axes avec des étiquettes explicites
    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.axvline(0, color='black', lw=0.8, ls='--')

    plt.title('Analyse en Composantes Multiples (ACM)', fontsize=14)
    plt.xlabel('Première composante principale', fontsize=12)
    plt.ylabel('Deuxième composante principale', fontsize=12)

    # Afficher le graphique
    plt.grid()
    plt.show()




def plot_residuals(y_true, y_pred):
    """
    Afficher le graphique des résidus en fonction des valeurs réelles.

    Parameters
    ----------
    y_true: array-like
        Les valeurs réelles.
    y_pred: array-like  
        Les valeurs prédites.

    Returns
    -------
    None
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, color='red', alpha=0.6)
    plt.axhline(0, color='blue', lw=2)
    plt.xlabel('Prix réel')
    plt.ylabel('Résidus')
    plt.title('Résidus vs Prix réel')
    plt.grid()
    plt.show()


