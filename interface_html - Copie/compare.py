import argparse
import pandas as pd

def compare_csv(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Exemple de comparaison simple: affichage des dimensions
    print(f"Taille du CSV 1: {df1.shape}")
    print(f"Taille du CSV 2: {df2.shape}")

    # Comparaison des colonnes
    common_columns = set(df1.columns).intersection(set(df2.columns))
    print(f"Colonnes communes: {common_columns}", flush=True)

    # Exemple de statistiques: nombre de lignes différentes
    differences = df1.compare(df2, keep_shape=True, keep_equal=False)
    print(f"Lignes différentes: {len(differences)}", flush=True )

    # Vous pouvez ajouter d'autres comparaisons/statistiques ici

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparer deux fichiers CSV')
    parser.add_argument('-file1', type=str, required=True, help='Chemin du premier fichier CSV')
    parser.add_argument('-file2', type=str, required=True, help='Chemin du deuxième fichier CSV')

    args = parser.parse_args()

    compare_csv(args.file1, args.file2)
