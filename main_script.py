import argparse
from config.config import SEED, TEST_RATIO
from src.data_processing.load_data import load_data
from src.data_science.data import data_process
from src.figures.figures_plot import plot_distributions, plot_categorical_distributions
from src.data_science.models import baseline_model, train_and_optimize_rf


def data_loading(filename):
    print(f"Loading data from {filename}...")
    load_data(filename)  # Passe l'argument `filename` à load_data()

def plot_distributions_function():
    print("Plotting distributions...")
    plot_distributions()

def plot_categorical_distributions_function():
    print("Plotting categorical distributions...")
    plot_categorical_distributions()

def data_process_function(data_path):
    print(f"Processing data from {data_path}...")
    data_process(data_path)  # Passer l'argument requis à la fonction data_process

def baseline_model_function():
    print("Calculating baseline model...")
    baseline_model()

def train_and_optimize_rf_function():
    print("Training and optimizing Random Forest...")
    train_and_optimize_rf()

def main():
    # Initialiser l'argument parser
    parser = argparse.ArgumentParser(description="Main script")
    
    # Ajouter les arguments disponibles pour la ligne de commande
    parser.add_argument("--step", type=str, required=True, choices=["data-load", "process", "baseline", "train", "plot"],
                        help="Specify the step to execute: data-load, process, baseline, train, or plot.")
    parser.add_argument("--data", type=str, help="Path to the data file", default="data.csv")  # Chemin par défaut pour les données
    parser.add_argument("--test_ratio", type=float, help="Test set ratio", default=0.2)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    
    # Parser les arguments fournis en ligne de commande
    args = parser.parse_args()

    # Exécuter la fonction associée à l'argument --step
    if args.step == "data-load":
        data_loading(args.data)  # Passer le chemin du fichier de données à load_data
    elif args.step == "process":
        data_process_function(args.data)  # Passer le chemin des données à data_process_function
    elif args.step == "baseline":
        baseline_model_function()
    elif args.step == "train":
        train_and_optimize_rf_function()
    elif args.step == "plot":
        plot_distributions_function()
    else:
        print("Invalid step. Please choose from: data-load, process, baseline, train, or plot.")

if __name__ == "__main__":
    main()
