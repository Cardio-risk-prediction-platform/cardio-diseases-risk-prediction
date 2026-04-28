Ce projet expose un modèle de Machine Learning via une API FastAPI permettant de prédire le risque de maladie cardiaque à partir de données patient.

L’API est ensuite connectable à une interface web (frontend).

Pour lancer le swagger, 
- créer un environnement virtuel
- installer des dépendances du fichier requirements.txt
- se placer dans le dossier model-training, exécuter le fichier .ipynb pour avoir le fichier cardio-model.pkl
- se placer dans le dossier model-api
- exécuter la commande python train.py
- lancer la commande uvicorn app:app --reload

L'url du swagger est accessible via http://127.0.0.1:8000/docs

Après avoir passé un fichier JSON, l'endpoint de prédiction est http://127.0.0.1:8000/predict

Reste plus qu'à la connecter au front !!
