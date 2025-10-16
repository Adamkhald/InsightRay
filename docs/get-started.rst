.. _get_started:

========================
Guide de Démarrage Rapide
========================

Ce guide fournit des instructions détaillées pour l'installation et le démarrage de chacun des projets d'intelligence artificielle médicale documentés.

Prérequis Généraux
==================

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre système :

* **Python (version recommandée : v3.7.x ou supérieure)** :
    * Vous pouvez télécharger Python depuis `python.org <https://www.python.org/downloads/>`_.
    * Il est recommandé d'utiliser un environnement virtuel (comme `venv` ou `conda`) pour gérer les dépendances de chaque projet.

Configuration de l'environnement virtuel :

.. code-block:: bash

    # Pour créer un environnement virtuel Python
    python -m venv env

    # Pour activer l'environnement virtuel (Linux/macOS)
    source env/bin/activate

    # Pour activer l'environnement virtuel (Windows)
    .\env\Scripts\activate

Installation des Dépendances
=============================

Chaque projet a ses propres dépendances listées dans un fichier `requirements.txt`. Activez votre environnement virtuel **avant** d'installer les dépendances pour un projet donné.

Guide d'Installation Projet par Projet
======================================

Ce qui suit décrit les étapes d'installation et de démarrage pour chaque projet :

Projet de Chatbot Médical
-------------------------

Ce projet développe un chatbot médical interactif basé sur Flask.

1.  **Structure du projet :**
    Assurez-vous que votre répertoire de projet est organisé comme suit :

    .. code-block::
        :force:

        medical_chatbot/
        ├── app.py
        ├── templates/
        │   └── index.html
        ├── requirements.txt
        ├── Medical_dataset/
        │   ├── intents_short.json
        │   ├── tfidfsymptoms.csv
        │   ├── Training.csv
        │   ├── symptom_Description.csv
        │   ├── symptom_severity.csv
        │   └── symptom_precaution.csv
        └── model/
            └── knn.pkl

    * `app.py` est le fichier principal de l'application Flask.
    * `templates/index.html` est le template HTML de l'interface utilisateur.
    * `Medical_dataset/` contient tous les fichiers de données nécessaires (assurez-vous d'y placer vos fichiers de données).
    * `model/knn.pkl` est votre modèle KNN pré-entraîné (placez-le ici).

2.  **Installation des dépendances :**
    Naviguez dans le répertoire `medical_chatbot` et installez les dépendances :

    .. code-block:: bash

        cd medical_chatbot
        pip install -r requirements.txt

    Les dépendances incluent `Flask`, `pandas`, `nltk`, `numpy`, `scikit-learn`, `joblib`, etc. NLTK téléchargera également des données supplémentaires lors de l'initialisation de l'application.

3.  **Exécution de l'application :**
    Lancez l'application Flask :

    .. code-block:: bash

        python app.py

    L'application sera accessible à `http://localhost:5000`.

Projet de Détection des Fractures Osseuses
------------------------------------------

Ce projet se concentre sur la détection et la classification automatisées des fractures osseuses à partir de radiographies.

1.  **Structure du projet :**
    Assurez-vous que les fichiers et répertoires sont organisés comme suit :

    .. code-block::
        :force:

        bone_fracture_project/
        ├── app.py                # Application Flask pour l'interface
        ├── predictions.py        # Logique de prédiction des modèles
        ├── prediction_test.py    # Script de test des prédictions
        ├── training_fracture.py  # Script d'entraînement des modèles de détection de fracture
        ├── training_parts.py     # Script d'entraînement du modèle de classification des parties osseuses
        ├── Dataset/              # Votre répertoire de données MURA
        │   └── (structure MURA: train/valid/body_part/patient_id/label/images)
        ├── weights/              # Répertoire pour les modèles entraînés (.h5 files)
        ├── plots/                # Répertoire pour les tracés générés
        │   └── BodyPartAcc.png
        │   └── BodyPartLoss.png
        │   └── FractureDetection/
        │       └── Elbow/
        │       └── Hand/
        │       └── Shoulder/
        ├── results/              # Répertoire pour les résultats de l'application
        ├── requirements.txt      # Dépendances Python
        └── readme.md             # Documentation du projet

2.  **Prérequis Python :**
    * Utilisez **Python v3.7.x**.

3.  **Installation des dépendances :**
    Naviguez dans le répertoire racine du projet et installez les dépendances :

    .. code-block:: bash

        pip install -r requirements.txt

    Les dépendances incluent `customtkinter`, `PyAutoGUI`, `PyGetWindow`, `Pillow`, `numpy`, `tensorflow`, `keras`, `pandas`, `matplotlib`, `scikit-learn`, `colorama`, `Flask`.

4.  **Pré-entraînement (facultatif mais recommandé) :**
    Pour entraîner les modèles (si les poids `ResNet50_BodyParts.h5`, `ResNet50_Elbow_frac.h5`, etc. ne sont pas déjà disponibles dans le dossier `weights/`), exécutez les scripts suivants :

    .. code-block:: bash

        python training_parts.py
        python training_fracture.py

    Ces scripts généreront et sauvegarderont les modèles dans le dossier `weights/` et les tracés dans le dossier `plots/`.

5.  **Exécution de l'application (Interface Graphique) :**
    Pour lancer l'interface web Flask permettant de tester la détection de fractures :

    .. code-block:: bash

        python app.py

Projet de Détection des Anomalies sur les Radiographies Pulmonaires
--------------------------------------------------------------

Ce projet implémente la détection d'objets sur les radiographies pulmonaires à l'aide de YOLOv8.

1.  **Structure du projet :**
    Votre répertoire de projet devrait inclure :

    .. code-block::
        :force:

        chest_xray_detection_project/
        ├── main_pipeline_script.py  # Script Python principal (celui que vous avez fourni)
        ├── train.csv                # Votre fichier CSV d'annotations
        ├── train/                   # Répertoire contenant les images de radiographies pulmonaires
        │   └── image_id_1.jpg
        │   └── image_id_2.jpg
        │   └── ...
        ├── test_images/             # Répertoire pour les images de test d'inférence (créez-le et ajoutez des images)
        ├── yolo_dataset/            # Répertoire de sortie pour le format YOLO (sera créé par le script)
        └── requirements.txt         # Dépendances Python (incluant ultralytics)

2.  **Installation des dépendances :**
    Créez un `requirements.txt` pour ce projet et installez les dépendances. Assurez-vous d'inclure `ultralytics`.

    Exemple de `requirements.txt` :

    .. code-block:: text

        pandas
        pyyaml
        scikit-learn
        matplotlib
        seaborn
        plotly
        Pillow
        ultralytics # Très important pour YOLOv8

    Installez-les :

    .. code-block:: bash

        pip install -r requirements.txt

3.  **Préparation des données au format YOLO :**
    Le script contient une fonction `convert_to_yolo_format` qui s'occupe de cette étape. Vous devrez exécuter la partie du script qui appelle cette fonction, en vous assurant que `CSV_FILE` et `IMAGES_DIR` sont correctement définis. Le script créera le répertoire `yolo_dataset/` et le fichier `dataset.yaml`.

    Si vous n'avez pas un script dédié pour cette étape, la partie suivante de votre code l'exécutera :

    .. code-block:: python

        # Dans votre script principal (par exemple, main_pipeline_script.py)
        # Assurez-vous que les variables de configuration sont définies
        CSV_FILE = 'train.csv'
        IMAGES_DIR = 'train'
        OUTPUT_DIR = 'yolo_dataset'

        # Chargez votre ensemble de données
        df = pd.read_csv(CSV_FILE)

        # Convertissez-le au format YOLO
        yaml_path, class_names = convert_to_yolo_format(df, IMAGES_DIR, OUTPUT_DIR)


4.  **Entraînement du modèle YOLOv8 :**
    Lancez l'entraînement du modèle via votre script Python principal. Assurez-vous que les variables de configuration (`MODEL_SIZE`, `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`, `DEVICE`) sont définies comme souhaité.

    .. code-block:: python

        # Dans votre script principal
        from ultralytics import YOLO

        model = YOLO(f"yolov8{MODEL_SIZE}.pt") # Charge un modèle pré-entraîné
        results = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            save=True,
            verbose=True
        )

5.  **Validation et Inférence :**
    Le script continuera à effectuer la validation et l'inférence. Placez les images que vous souhaitez tester dans le répertoire `test_images/`.

    .. code-block:: python

        # Dans votre script principal
        # Validation
        model = YOLO('best.pt') # Charge le meilleur modèle entraîné
        val_results = model.val(data=yaml_path)

        # Inférence
        TEST_DIR = 'test_images'
        if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
            results = model.predict(
                source=TEST_DIR,
                conf=0.25,
                save=True
            )
            # ... (visualisation des résultats) ...
        else:
            print(f"Le répertoire de test {TEST_DIR} est vide ou n'existe pas. Veuillez y ajouter des images.")
