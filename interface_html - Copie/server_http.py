import requests
import json
import time

def update_value(prediction):
    # Utiliser une session pour des connexions persistantes
    session = requests.Session()

    # Définir l'URL de l'IP pour la requête GET
    url_get = "http://192.168.127.254/api/slot/0/io/do/0/doStatus"  # Remplacez par l'IP et le chemin appropriés

    # Envoyer la requête GET
    response_get = requests.get(url_get, headers={"Content-Type": "application/json", "Accept": "vdn.dac.v1"})

    # Vérifier si la requête GET a réussi
    if response_get.status_code == 200:
        t0 = time.time()
        # Récupérer le corps de la réponse et le convertir en dictionnaire
        try:
            body_dict = response_get.json()

        except json.JSONDecodeError:
            print("Erreur: La réponse GET n'est pas un JSON valide.")
            body_dict = None

        if body_dict is not None:
            # Mettre à jour la valeur spécifique
            body_dict["io"]["do"]['0']["doStatus"] = prediction  # Remplacez <nouvelle_valeur> par la valeur souhaitée

            # Convertir le dictionnaire mis à jour en chaîne JSON
            body_json = json.dumps(body_dict)

            # Définir l'URL pour la requête PUT
            url_put = "http://192.168.127.254/api/slot/0/io/do/0/doStatus"  # Remplacez par l'IP et le chemin appropriés

            # Envoyer la requête PUT avec le corps mis à jour
            headers = {'Content-Type': 'application/json',  "Accept": "vdn.dac.v1"}
            response_put = session.put(url_put, data=body_json, headers=headers)

            # Vérifier si la requête PUT a réussi
            """if response_put.status_code == 200:
                print("La requête PUT a réussi en ", time.time()-t0, "ms")
            else:
                print(f"Erreur lors de la requête PUT: {response_put.status_code}")"""
    else:
        print(f"Erreur lors de la requête GET: {response_get.status_code}")

    # Fermer la session
    session.close()


if __name__ == '__main__':

    update_value()