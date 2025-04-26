import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def baseline_model(data):
    """
    Calculer le prix moyen par nombre de chambres.
    
    Parameters
    ----------
    data: DataFrame
        Les données d'entraînement.
    
    Returns
    -------
    DataFrame
        Le prix moyen par nombre de chambres.
    """
    # Calculer le prix moyen par nombre de chambres
    baseline = data.groupby('chambres')['prix'].mean()
    
    return baseline

def plot_predictions_baseline(y_test, y_pred, title):
    """ Afficher le nuage de points des prédictions et des valeurs réelles avec la droite de référence y=x.

    Args: 
        y_test (pd.Series): Les valeurs réelles.
        y_pred (pd.Series): Les valeurs prédites.
        title (str): Le titre du graphique.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='salmon')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Ligne de référence y=x
    plt.xlabel('Prix réels')
    plt.ylabel('Prédictions')
    plt.title(title)
    plt.show() 


def calcul_performance(max_depth_values, n_estimators_values, X_train, y_train, X_test, y_test, SEED):
    """
    Calculer les performances du modèle Random Forest en fonction de max_depth et n_estimators.
    
    Parameters
    ----------
    max_depth_values: list
        Liste des valeurs de max_depth à tester.
    n_estimators_values: list
        Liste des valeurs de n_estimators à tester.
    X_train: array-like
        Features d'entraînement.
    y_train: array-like
        Cibles d'entraînement.
    X_test: array-like
        Features de test.
    y_test: array-like  
        Cibles de test.
    SEED: int
        Aléa pour garantir la reproductibilité.

    Returns
    -------
    train_rmse_depth: list
        RMSE sur les données d'entraînement en fonction de max_depth.
    test_rmse_depth: list
        RMSE sur les données de test en fonction de max_depth.
    train_rmse_estimators: list
        RMSE sur les données d'entraînement en fonction de n_estimators.
    test_rmse_estimators: list
        RMSE sur les données de test en fonction de n_estimators.
    """
    train_rmse_depth = []
    test_rmse_depth = []
    train_rmse_estimators = []
    test_rmse_estimators = []

    # Évaluer la performance en fonction de max_depth
    for max_depth in max_depth_values:
        rf_model = RandomForestRegressor(max_depth=max_depth, n_estimators=100, random_state=SEED)
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        train_rmse_depth.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse_depth.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # Évaluer la performance en fonction de n_estimators
    for n_estimators in n_estimators_values:
        rf_model = RandomForestRegressor(max_depth=20, n_estimators=n_estimators, random_state=SEED)
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        train_rmse_estimators.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse_estimators.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    return train_rmse_depth, test_rmse_depth, train_rmse_estimators, test_rmse_estimators






def train_and_optimize_rf(X_train, y_train, X_test, y_test, seed):
    """
    Entraîne un modèle Random Forest, optimise les hyperparamètres avec GridSearchCV,
    et évalue ses performances sur des données de test.

    Parameters:
    - X_train: array-like, Features d'entraînement
    - y_train: array-like, Cibles d'entraînement
    - X_test: array-like, Features de test
    - y_test: array-like, Cibles de test
    - seed: int, Aléa pour garantir la reproductibilité

    Returns:
    - best_rf_model: Modèle Random Forest optimisé
    - best_params: Meilleurs paramètres trouvés lors de la recherche
    - evaluation_metrics: Dictionnaire contenant MAE, RMSE et R²
    """
    # Définir le modèle de base sans hyperparamètres spécifiques
    rf_model = RandomForestRegressor(random_state=seed)

    # Définir la grille de paramètres pour GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],       # Nombre d'arbres dans la forêt
        'max_depth': [10, 20, None],          # Profondeur maximale des arbres
        'min_samples_split': [2, 5, 10]       # Nombre minimal d'échantillons pour diviser un nœud
    }

    # Configurer GridSearchCV avec 5 sous-ensembles pour la validation croisée
    grid_search = GridSearchCV(estimator=rf_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='neg_mean_squared_error', 
                               n_jobs=-1)

    # Effectuer la recherche sur la meilleure combinaison de paramètres
    print("Optimisation des hyperparamètres en cours...")
    grid_search.fit(X_train, y_train)
    print("Optimisation terminée.")

    # Extraire le modèle avec les meilleurs paramètres
    best_rf_model = grid_search.best_estimator_

    # Prédire les prix sur le jeu de test avec le meilleur modèle trouvé
    y_pred = best_rf_model.predict(X_test)

    # Calculer le MAE, RMSE et le R²
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Stocker les métriques dans un dictionnaire
    evaluation_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    # Extraire les meilleurs paramètres
    best_params = grid_search.best_params_

    return {
        'best_rf_model': best_rf_model, 
        'best_params': best_params, 
        'evaluation_metrics': evaluation_metrics
    }

