import argparse
import shutil
import time
import os
from glob import glob

# Analyse des arguments
parser = argparse.ArgumentParser(description='Simuler le traitement d\'images.')
parser.add_argument('-source_dir', type=str, help='Dossier source contenant les images')
parser.add_argument('-dest_dir', type=str, help='Dossier de destination pour les images copiées')
args = parser.parse_args()

source_dir = args.source_dir
dest_dir = args.dest_dir

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Liste des fichiers image
image_files = glob(os.path.join(source_dir, '*'))

# Stocker les temps d'exécution
times = []

compteur = 0

for image_file in image_files:
    


    start_time = time.time()
    shutil.copy(image_file, dest_dir)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Temps en millisecondes
    times.append(elapsed_time)

    #print(f'Temps d\'exécution pour itération {compteur + 1}: {elapsed_time:.2f} ms', flush=True)
    compteur += 1
    if compteur %3 == 0 and compteur !=0:
        time.sleep(0.3)

    if compteur == 30:

        break 