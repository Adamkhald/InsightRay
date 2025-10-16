.. _medical_ai_projects:

=====================================
Documentation Détaillée des Projets d'IA Médicale
=====================================

.. contents:: Table des matières
   :depth: 2
   :local:

Introduction Générale
=====================
Ce document combine la documentation de trois projets distincts d'intelligence artificielle dans le domaine médical : la détection des fractures osseuses, la détection des anomalies sur les radiographies pulmonaires, et un chatbot médical interactif. Chaque projet vise à améliorer le diagnostic et le traitement des patients grâce à l'application de techniques d'apprentissage automatique et d'apprentissage profond, en offrant des solutions complémentaires aux méthodes diagnostiques traditionnelles.

===================================
Projet de Détection des Anomalies sur les Radiographies Pulmonaires
===================================

.. contents:: Table des matières du Projet Radiographie Pulmonaire
   :depth: 2
   :local:

Introduction
============
Ce projet met en œuvre un pipeline de détection d'objets pour les images de radiographies pulmonaires, se concentrant principalement sur l'identification de diverses anomalies pathologiques. Il exploite le cadre de pointe **YOLOv8 (You Only Look Once, version 8)** pour une détection efficace et précise des anomalies. Le pipeline comprend un prétraitement complet des données, l'entraînement du modèle avec transfert d'apprentissage, la validation et la démonstration d'inférence.

Données
====

.. figure:: images/chest_xray_data_example.png
   :alt: Exemple de radiographie pulmonaire avec annotations
   :width: 700px

   Un exemple d'image de radiographie pulmonaire illustrant les annotations de boîtes englobantes pour diverses anomalies dans l'ensemble de données.

Le projet repose sur un ensemble de données d'images de radiographies pulmonaires accompagnées d'annotations de boîtes englobantes. L'entrée principale des données est un fichier CSV contenant les métadonnées et les détails d'annotation, ainsi qu'un répertoire d'images correspondantes.

Structure de l'ensemble de données :
------------------
L'ensemble de données devrait contenir :

* **`train.csv`**: Un fichier CSV contenant des annotations avec des colonnes telles que :
    * `image_id`: Identifiant unique pour chaque image.
    * `class_name`: Le nom de la classe détectée (par exemple, "Pneumonia", "Cardiomegaly", "No finding").
    * `class_id`: ID numérique correspondant à `class_name`.
    * `x_min`, `y_min`, `x_max`, `y_max`: Coordonnées des boîtes englobantes (valeurs en pixels) pour l'anomalie détectée.
* **`train/` directory**: Contient les fichiers d'images de radiographies pulmonaires réels (par exemple, `image_id.jpg`).

Caractéristiques des données :
---------------------
* **Distribution des classes :** L'ensemble de données présente généralement une distribution de classes déséquilibrée, avec "No finding" étant souvent la classe majoritaire. Les autres classes représentent des pathologies spécifiques.
* **Images multi-étiquettes :** De nombreuses radiographies pulmonaires peuvent présenter plusieurs anomalies, ce qui signifie qu'un seul `image_id` peut avoir plusieurs lignes dans le CSV, chacune correspondant à une boîte englobante et un `class_name` différents.

Étapes de prétraitement des données :
-------------------------
Avant d'entraîner le modèle YOLOv8, les données subissent plusieurs étapes de prétraitement cruciales :

1.  **Suppression non maximale (NMS) :**
    * **Objectif :** Éliminer les annotations de boîtes englobantes redondantes ou fortement superposées pour le même objet au sein d'une image. Cela est crucial car les ensembles de données peuvent parfois contenir plusieurs annotations légèrement différentes pour la même anomalie.
    * **Méthode :** Une métrique d'Intersection sur Union (IoU) est utilisée pour mesurer le chevauchement. Si deux boîtes pour la même classe ont un IoU supérieur à un certain seuil (par exemple, 0,5), l'une est supprimée.
2.  **Suppression des très petites boîtes englobantes :**
    * **Objectif :** Filtrer le bruit ou les annotations insignifiantes qui sont trop petites pour être significatives ou apprenables par le modèle.
    * **Méthode :** Les boîtes englobantes avec une largeur ou une hauteur inférieure à une `MIN_BOX_SIZE` prédéfinie (par exemple, 10 pixels) sont supprimées.
3.  **Sous-échantillonnage de la classe majoritaire (`No finding`) :**
    * **Objectif :** Gérer le déséquilibre des classes, en particulier lorsqu'une classe "No finding" (images sans pathologie détectée) est significativement surreprésentée.
    * **Méthode :** Un sous-ensemble aléatoire des exemples "No finding" est échantillonné pour ramener sa proportion plus près des autres classes, empêchant le modèle de devenir biaisé vers la prédiction de "No finding".
4.  **Suppression des classes minoritaires :**
    * **Objectif :** Supprimer les classes qui ont un nombre insuffisant d'échantillons (par exemple, moins de 500 annotations). L'entraînement sur des classes très clairsemées peut entraîner de mauvaises performances et une instabilité du modèle.
    * **Méthode :** Les classes tombant en dessous d'un seuil `min_samples` sont exclues de l'ensemble de données.

Modèles
======
Le composant central pour la détection d'objets dans ce projet est le modèle **YOLOv8**.

Modèle YOLOv8 :
-------------
* **Type :** YOLO (You Only Look Once), version 8, un modèle avancé de détection d'objets en une seule étape.
* **Architecture :** YOLOv8 est conçu pour la détection d'objets en temps réel, prédisant simultanément les coordonnées des boîtes englobantes et les probabilités de classe. Il existe en différentes tailles (nano, petit, moyen, grand, extra-large) offrant un compromis entre vitesse et précision. Le projet utilise une `MODEL_SIZE` configurable (par exemple, 'm' pour moyen).
* **Transfert d'apprentissage :** Le modèle est initialisé avec des poids pré-entraînés à partir d'un grand ensemble de données (généralement COCO), ce qui lui permet de tirer parti des connaissances préexistantes des caractéristiques visuelles courantes. Cela réduit considérablement le temps d'entraînement et améliore les performances sur des ensembles de données plus petits et spécialisés comme les images médicales.
* **Entrée :** Coordonnées de boîtes englobantes normalisées et ID de classe ainsi que des données d'image.
* **Sortie :** Pour une image donnée, le modèle produit une liste d'objets détectés, chacun avec :
    * Les coordonnées des boîtes englobantes (`x_min`, `y_min`, `x_max`, `y_max`).
    * Un score de confiance (à quel point le modèle est sûr de la détection).
    * Un `class_id` prédit (et le `class_name` correspondant).

Processus
=======
L'ensemble du pipeline, de la préparation des données à l'inférence du modèle, est orchestré via un script Python conçu pour une utilisation facile et une reproductibilité.

1.  **Définition de la configuration :**
    * Des paramètres clés tels que `CSV_FILE`, `IMAGES_DIR`, `OUTPUT_DIR`, `MODEL_SIZE`, `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE` et `DEVICE` sont définis au début du script. Ceux-ci peuvent être facilement modifiés pour s'adapter à différents ensembles de données ou exigences d'entraînement.

    .. code-block:: python

        # Extrait du script principal
        CSV_FILE = 'train.csv'
        IMAGES_DIR = 'train'
        OUTPUT_DIR = 'yolo_dataset'
        MODEL_SIZE = 'm'
        EPOCHS = 7
        # ... et d'autres configurations

2.  **Chargement et inspection initiale de l'ensemble de données :**
    * Le fichier CSV d'annotation brut est chargé dans un DataFrame Pandas.
    * Des statistiques de base sur l'ensemble de données, y compris la forme, les images uniques, les classes uniques et la distribution initiale des classes, sont affichées à l'utilisateur.

3.  **Conversion au format YOLO (`convert_to_yolo_format` fonction) :**
    * **Configuration du répertoire :** Crée une structure de répertoire pour les données YOLOv8 (`yolo_dataset/images/train`, `yolo_dataset/images/val`, `yolo_dataset/labels/train`, `yolo_dataset/labels/val`).
    * **Mappage des classes :** Détermine tous les noms de classes uniques à partir de l'ensemble de données et les mappe à des ID entiers (indexés à partir de 0). Ce mappage est enregistré dans un fichier `classes.txt`.
    * **Division train/validation :** Divise les `image_id` en ensembles d'entraînement (par exemple, 80 %) et de validation (par exemple, 20 %), garantissant que toutes les annotations pour une image restent dans le même ensemble.
    * **Génération d'images et d'étiquettes :** Pour chaque image :
        * Le fichier image est copié dans le sous-répertoire `images/train` ou `images/val` approprié dans `OUTPUT_DIR`.
        * Un fichier d'étiquette `.txt` correspondant est créé dans `labels/train` ou `labels/val`. Ce fichier contient une ligne par boîte englobante, formatée comme `class_id x_center y_center width height` (toutes normalisées à une plage de 0-1 par rapport aux dimensions de l'image).
    * **Configuration YAML :** Un fichier `dataset.yaml` est généré, qui sert de configuration centrale pour YOLOv8, pointant vers les répertoires d'images et d'étiquettes, le nombre de classes et les noms de classes.

    .. code-block:: python

        # Extrait de la fonction convert_to_yolo_format
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        # ... (création d'autres répertoires) ...
        # ... (mappage de classes et division de fichiers) ...
        # ... (copie d'images et écriture de fichiers d'étiquettes) ...
        # ... (génération de dataset.yaml) ...

4.  **Entraînement du modèle YOLO :**
    * Un modèle `ultralytics.YOLO` est initialisé, généralement en chargeant un modèle YOLOv8 pré-entraîné (`yolov8{MODEL_SIZE}.pt`).
    * La méthode `model.train()` est appelée, en utilisant `dataset.yaml` pour la configuration des données et les `epochs`, `batch_size`, `img_size` et `device` spécifiés pour les paramètres d'entraînement.
    * Le processus d'entraînement enregistre les points de contrôle et les poids du meilleur modèle (`best.pt`).

    .. code-block:: python

        # Extrait du script principal
        model = YOLO(f"yolov8{MODEL_SIZE}.pt")
        results = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            save=True,
            verbose=True
        )

5.  **Validation du modèle :**
    * Le meilleur modèle (`best.pt`, celui avec les meilleures performances de validation) est chargé.
    * `model.val()` est exécuté sur l'ensemble de validation spécifié dans `dataset.yaml` pour calculer les métriques de performance (par exemple, mAP) du modèle entraîné.

    .. code-block:: python

        # Extrait du script principal
        model = YOLO('best.pt') # Charger le meilleur modèle entraîné
        val_results = model.val(data=yaml_path)

6.  **Inférence sur les images de test :**
    * Le modèle entraîné peut être utilisé pour effectuer des prédictions sur de nouvelles images non vues situées dans un `TEST_DIR`.
    * `model.predict()` exécute la détection sur ces images, en appliquant un seuil de confiance pour filtrer les résultats.
    * Les boîtes englobantes détectées et les étiquettes de classe sont enregistrées sur les images.
    * Une démonstration visuelle montre quelques exemples de résultats d'inférence avec des annotations superposées sur les images originales.

    .. code-block:: python

        # Extrait du script principal
        results = model.predict(
            source=TEST_DIR,
            conf=0.25,
            save=True
        )
        # ... (visualisation des résultats) ...

Résultats
=======
Le résultat de ce pipeline est un modèle YOLOv8 entraîné capable de détecter diverses anomalies sur les images de radiographies pulmonaires, ainsi que des métriques de performance détaillées et des démonstrations visuelles de ses capacités.

Sorties attendues :
-----------------
* **Poids du modèle entraîné :** Un fichier `best.pt` contenant les poids optimisés du modèle YOLOv8, prêt pour le déploiement.
* **Métriques d'entraînement :** Journaux et tracés (générés par YOLOv8) montrant la progression de l'entraînement (perte, précision, mAP) sur les époques.
* **Métriques de validation :** Métriques de performance quantitatives (par exemple, précision, rappel, mAP) sur l'ensemble de données de validation.
* **Résultats d'inférence :** Images avec des boîtes englobantes superposées et des étiquettes de classe pour les anomalies détectées, démontrant les performances du modèle sur des données non vues.

Exemple d'interface de détection de radiographie pulmonaire :
----------------------------------------
Bien qu'une interface utilisateur graphique (GUI) dédiée à l'inférence de détection de radiographie pulmonaire ne fasse pas explicitement partie du script fourni, dans une application réelle, les résultats de l'inférence seraient généralement affichés via une interface similaire à cet exemple conceptuel :

.. figure:: images/GUI/chest_xray_detection_interface.png
   :alt: Interface conceptuelle de détection de radiographie pulmonaire
   :width: 800px

   Une image conceptuelle d'une interface utilisateur où les images de radiographies pulmonaires sont téléchargées, et les anomalies détectées (avec des boîtes englobantes et des étiquettes) sont affichées à l'utilisateur. Cela inclurait généralement des scores de confiance et potentiellement une liste de pathologies identifiées.

=============================
Projet de Détection des Fractures Osseuses
=============================

.. contents:: Table des matières du Projet Fracture Osseuse
   :depth: 2
   :local:

Introduction
============
Ce projet se concentre sur la détection et la classification automatisées des fractures osseuses à partir d'images de radiographies à l'aide de techniques d'apprentissage profond. L'objectif principal est d'améliorer la précision et l'efficacité du diagnostic des fractures, en fournissant une solution robuste pour compléter ou améliorer les méthodes de diagnostic traditionnelles dépendantes de l'homme.

Données
====
Le projet utilise l'ensemble de données **MURA (Musculoskeletal Radiographs)**, une collection complète d'images de radiographies musculo-squelettiques.

Aperçu de l'ensemble de données :
------------------
L'ensemble de données MURA comprend 20 335 images de radiographies, classées en trois parties osseuses distinctes :

.. list-table:: Distribution de l'ensemble de données MURA
   :widths: 20 20 20 20
   :header-rows: 1

   * - Partie
     - Normale
     - Fracturée
     - Total
   * - Coude
     - 3160
     - 2236
     - 5396
   * - Main
     - 4330
     - 1673
     - 6003
   * - Épaule
     - 4496
     - 4440
     - 8936

Structure des données :
---------------
L'ensemble de données est organisé en ensembles `train` et `valid`. Chaque ensemble contient des dossiers spécifiques au patient, et dans chaque dossier de patient, il y a généralement 1 à 3 images de radiographies correspondant à la même partie osseuse.

Division des données :
---------------
Pour l'entraînement et l'évaluation du modèle, l'ensemble de données est divisé comme suit :

* **Ensemble d'entraînement :** 72 % des données
* **Ensemble de validation :** 18 % des données
* **Ensemble de test :** 10 % des données

Modèles
======
Le cœur de ce projet repose sur les Réseaux de Neurones Convolutifs (CNN), en tirant spécifiquement parti de l'architecture **ResNet50**. La solution utilise une approche de classification en deux étapes, nécessitant deux types de modèles :

1.  **Modèle de classification des parties osseuses :**
    * **Objectif :** Identifier le type d'os spécifique (Coude, Main ou Épaule) présent dans une image de radiographie en entrée.
    * **Architecture :** Basée sur un modèle **ResNet50** pré-entraîné (sans sa couche de classification supérieure), suivi de couches denses personnalisées adaptées à la classification en 3 catégories.
    * **Pré-entraînement :** Le modèle de base ResNet50 utilise des poids pré-entraînés sur l'ensemble de données ImageNet, et ses couches sont initialement figées (`trainable = False`) pour agir comme un extracteur de caractéristiques.

2.  **Modèles de détection des fractures (spécifiques à la partie) :**
    * **Objectif :** Déterminer si une partie osseuse détectée est fracturée ou normale. Il existe un modèle distinct pour chaque type d'os.
    * **Architecture :** Trois modèles distincts, chacun utilisant une base **ResNet50**, sont entraînés. Chaque modèle est spécialisé pour l'un des trois types d'os (Coude, Main, Épaule).
    * **Sortie :** Chaque modèle classe son image d'os respective dans l'une des deux catégories : 'fracturée' ou 'normale'.

Entraînement
========
Le processus d'entraînement implique deux phases distinctes, correspondant aux deux types de modèles utilisés dans le pipeline de classification.

Aspects communs de l'entraînement :
-------------------------
* **Augmentation des données :** Des techniques telles que le retournement horizontal sont appliquées aux images d'entraînement pour augmenter la diversité de l'ensemble de données et améliorer la généralisation du modèle.
* **Prétraitement :** Les images sont prétraitées à l'aide de `tf.keras.applications.resnet50.preprocess_input` pour correspondre aux exigences d'entrée du modèle ResNet50.
* **Dimensions de l'image :** Toutes les images sont redimensionnées à 224x224 pixels avec 3 canaux RVB.
* **Optimiseur :** L'optimiseur Adam avec un faible taux d'apprentissage (0.0001) est utilisé pour l'entraînement.
* **Fonction de perte :** `categorical_crossentropy` est employée comme fonction de perte.
* **Callbacks :** L'arrêt précoce (Early Stopping) est utilisé pour surveiller la perte de validation et prévenir le surapprentissage, en restaurant les meilleurs poids trouvés pendant l'entraînement.

Entraînement du modèle de classification des parties osseuses :
-----------------------------------------
Ce modèle est entraîné pour distinguer les radiographies du coude, de la main et de l'épaule.

.. code-block:: python

    # Extrait de training_parts.py pour contexte
    # Compilation et entraînement du modèle pour la classification des parties osseuses
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25,
                        callbacks=[callbacks])
    model.save(THIS_FOLDER + "/weights/ResNet50_BodyParts.h5")

**Tracés de perte et de précision pour la prédiction des parties du corps :**

.. figure:: BodyPartAcc.png
   :alt: Tracé de précision des parties du corps

   Tracé de précision pour le modèle de prédiction des parties du corps.

.. figure:: BodyPartLoss.png
   :alt: Tracé de perte des parties du corps

   Tracé de perte pour le modèle de prédiction des parties du corps.


Entraînement des modèles de détection des fractures (spécifiques à la partie) :
---------------------------------------------------
Des modèles distincts sont entraînés pour chaque partie osseuse (Coude, Main, Épaule) afin de détecter la présence d'une fracture.

**Modèle de détection des fractures du coude :**

.. code-block:: python

    # Extrait de training_fracture.py pour contexte (exemple pour le coude)
    # Compilation et entraînement du modèle pour la détection des fractures du coude
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])
    model.save(THIS_FOLDER + "/weights/ResNet50_Elbow_frac.h5")

**Tracés de perte et de précision pour la détection des fractures du coude :**

.. figure:: FractureDetection/Elbow/_Accuracy.jpeg
   :alt: Tracé de précision de la détection des fractures du coude

   Tracé de précision pour le modèle de détection des fractures du coude.

.. figure:: FractureDetection/Elbow/_Loss.jpeg
   :alt: Tracé de perte de la détection des fractures du coude

   Tracé de perte pour le modèle de détection des fractures du coude.

**Modèle de détection des fractures de la main :**

.. code-block:: python

    # Extrait de training_fracture.py pour contexte (exemple pour la main)
    # Compilation et entraînement du modèle pour la détection des fractures de la main
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])
    model.save(THIS_FOLDER + "/weights/ResNet50_Hand_frac.h5")

**Tracés de perte et de précision pour la détection des fractures de la main :**

.. figure:: FractureDetection/Hand/_Accuracy.jpeg
   :alt: Tracé de précision de la détection des fractures de la main

   Tracé de précision pour le modèle de détection des fractures de la main.

.. figure:: FractureDetection/Hand/_Loss.jpeg
   :alt: Tracé de perte de la détection des fractures de la main

   Tracé de perte pour le modèle de détection des fractures de la main.

**Modèle de détection des fractures de l'épaule :**

.. code-block:: python

    # Extrait de training_fracture.py pour contexte (exemple pour l'épaule)
    # Compilation et entraînement du modèle pour la détection des fractures de l'épaule
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])
    model.save(THIS_FOLDER + "/weights/ResNet50_Shoulder_frac.h5")

**Tracés de perte et de précision pour la détection des fractures de l'épaule :**

.. figure:: FractureDetection/Shoulder/_Accuracy.jpeg
   :alt: Tracé de précision de la détection des fractures de l'épaule

   Tracé de précision pour le modèle de détection des fractures de l'épaule.

.. figure:: FractureDetection/Shoulder/_Loss.jpeg
   :alt: Tracé de perte de la détection des fractures de l'épaule

   Tracé de perte pour le modèle de détection des fractures de l'épaule.

Pipeline d'architecture de classification
====================================

L'architecture globale du pipeline de détection des fractures osseuses implique un processus en deux étapes pour identifier d'abord la partie osseuse, puis détecter une fracture dans cette partie spécifique.

.. figure:: images/Architecture.png
   :alt: Architecture de l'algorithme

   Diagramme illustrant le pipeline d'architecture de classification en deux étapes.

=====================================
Projet de Chatbot Médical
=====================================

.. contents:: Table des matières du Projet Chatbot Médical
   :depth: 2
   :local:

Introduction
============
Ce projet développe un chatbot médical interactif utilisant Flask et l'apprentissage automatique. Sa fonction principale est de diagnostiquer les maladies potentielles en fonction des symptômes fournis par l'utilisateur, d'offrir des informations pertinentes telles que des descriptions de maladies et des précautions, et d'évaluer la gravité des symptômes. Le chatbot utilise le Traitement du Langage Naturel (TLN) pour comprendre l'entrée de l'utilisateur et un modèle K-Nearest Neighbors (KNN) pour la prédiction des maladies.

Données
====

.. figure:: images/medical_data_overview.png
   :alt: Aperçu des fichiers de données médicales
   :width: 600px

   Un diagramme illustratif montrant l'interconnexion des divers fichiers de données médicales utilisés dans le projet.

Le chatbot exploite plusieurs ensembles de données structurées pour son fonctionnement, principalement situés dans le répertoire `Medical_dataset/`. Ces fichiers fournissent la base de connaissances pour l'identification des symptômes, la prédiction des maladies et la fourniture d'informations médicales pertinentes.

Fichiers de l'ensemble de données :
----------------
* **`Training.csv`**:
    * **Objectif :** Le principal ensemble de données utilisé pour l'entraînement du modèle de prédiction des maladies. Il contient des lignes représentant des cas de patients, avec des colonnes pour divers symptômes (indicateurs binaires de présence/absence) et une colonne finale pour le `prognosis` (maladie) diagnostiqué.
    * **Rôle :** Fournit la cartographie principale symptôme-maladie pour le modèle d'apprentissage automatique.
* **`Testing.csv`**:
    * **Objectif :** Utilisé pour évaluer les performances du modèle de prédiction des maladies entraîné. Il a la même structure que `Training.csv`.
    * **Rôle :** Garantit que les prédictions du modèle sont précises sur des données non vues.
* **`tfidfsymptoms.csv`**:
    * **Objectif :** Contient des représentations vectorisées TF-IDF (Term Frequency-Inverse Document Frequency) des descriptions de symptômes.
    * **Rôle :** Crucial pour le composant NLP, permettant au chatbot de comparer les descriptions de symptômes saisies par l'utilisateur avec des symptômes connus à l'aide de la similarité cosinus pour les identifier avec précision.
* **`intents_short.json`**:
    * **Objectif :** Définit la compréhension du chatbot des diverses "intentions" médicales et des modèles de symptômes associés.
    * **Rôle :** Aide à mapper les descriptions de symptômes en langage naturel à des noms de symptômes standardisés reconnus par le système.
* **`symptom_Description.csv`**:
    * **Objectif :** Mappe chaque maladie à une description concise.
    * **Rôle :** Utilisé pour fournir aux utilisateurs un contexte informatif sur leur diagnostic prédit.
* **`symptom_severity.csv`**:
    * **Objectif :** Attribue un score de gravité à des symptômes individuels.
    * **Rôle :** Utilisé conjointement avec la durée des symptômes pour calculer un niveau de gravité global pour l'état du patient.
* **`symptom_precaution.csv`**:
    * **Objectif :** Mappe les maladies à une liste de précautions recommandées.
    * **Rôle :** Offre des conseils pratiques aux utilisateurs en fonction de leur diagnostic prédit.

Modèle
=====
Le cœur de la capacité prédictive du chatbot est un classifieur **K-Nearest Neighbors (KNN)**.

Modèle de prédiction des maladies :
-------------------------
* **Type :** Classifieur K-Nearest Neighbors (KNN).
* **Entrée des données d'entraînement :** Vecteurs de symptômes codés en un seul coup (OHV). Chaque caractéristique du vecteur OHV correspond à un symptôme spécifique, avec une valeur de 1 si le symptôme est présent et 0 sinon.
* **Sortie des données d'entraînement :** Le `prognosis` (maladie) associé à la combinaison de symptômes.
* **Rôle :** Une fois que les symptômes de l'utilisateur sont convertis au format OHV, le modèle KNN identifie les `k` cas de patients historiques les plus similaires (voisins) et prédit la maladie en fonction de la classe majoritaire parmi ces voisins.
* **Implémentation :** Le modèle KNN entraîné est sérialisé et chargé à partir de `model/knn.pkl`.

Modèles/Techniques NLP de support :
----------------------------------
Bien qu'il ne s'agisse pas de "modèles" distincts au sens du classifieur KNN, plusieurs techniques NLP et structures de données prétraitées agissent comme des modèles de support pour la compréhension du langage naturel :

* **Vectoriseur TF-IDF :** Utilisé pour convertir les descriptions textuelles des symptômes en vecteurs numériques (`tfidfsymptoms.csv`). Cela permet une comparaison sémantique entre l'entrée de l'utilisateur et les symptômes connus.
* **Similarité Cosinus :** Employée pour calculer la similarité entre le vecteur TF-IDF de la description d'un symptôme de l'utilisateur et les vecteurs TF-IDF des symptômes connus, aidant à identifier avec précision le symptôme.
* **Composants NLTK :**
    * **Tokenisation (tokeniseur Punkt) :** Décompose le texte en mots individuels.
    * **Lemmatisation (Lemmatiseur WordNet) :** Réduit les mots à leur forme de base (par exemple, "courir" pour "running").
    * **Suppression des mots vides :** Élimine les mots courants (par exemple, "le", "est") qui ne portent pas de signification significative pour l'identification des symptômes.

Processus
=======
Le chatbot fonctionne comme une application web Flask, guidant l'utilisateur à travers une conversation structurée pour recueillir les symptômes et fournir un diagnostic. L'ensemble du processus est basé sur l'état, géré par un dictionnaire `chat_state` dans la session de l'utilisateur.

1.  **Initialisation de l'application (`initialize_app`) :**
    * Au démarrage de l'application, tous les composants nécessaires sont chargés : le modèle KNN (`knn.pkl`), divers ensembles de données médicales (`.csv` et `.json`), et les données NLTK.
    * Les dictionnaires globaux (`severityDictionary`, `description_list`, `precautionDictionary`) sont remplis pour une consultation rapide pendant les conversations.

    .. code-block:: python

        # Extrait de app.py
        def initialize_app():
            if not load_model_and_data():
                print("Failed to load necessary data. Some features may not work.")
                return False
            getSeverityDict()
            getprecautionDict()
            getDescription()
            return True

2.  **Début de la conversation et gestion de l'état :**
    * Lorsqu'un utilisateur accède à l'URL racine (`/`), une nouvelle session de discussion est initialisée, définissant le `chat_state` par défaut (`'name'` étape).
    * Tous les messages de l'utilisateur sont envoyés au point d'API `/chat`, où la fonction `process_chat_message` dirige le flux de conversation en fonction de `chat_state['step']`.

    .. code-block:: python

        # Extrait de app.py
        def init_chat_state():
            return {
                'step': 'name',
                'name': '',
                'symptoms': [],
                'current_symptom_options': [],
                'current_symptom_index': 0,
                'possible_diseases': [],
                'additional_symptoms_asked': [],
                'current_disease_index': 0,
                'current_disease_symptom_index': 0,
                'awaiting_days': False,
                'final_diagnosis': None
            }

        @app.route('/')
        def index():
            if 'chat_state' not in session:
                session['chat_state'] = init_chat_state()
            return render_template('index.html')

        @app.route('/chat', methods=['POST'])
        def chat():
            # ... (logique de traitement des messages) ...
            response = process_chat_message(user_message, chat_state)
            # ... (mise à jour de la session et retour de la réponse) ...

3.  **Identification et confirmation des symptômes :**
    * Le chatbot demande d'abord le nom de l'utilisateur, puis son symptôme principal.
    * Il utilise `predictSym()` qui applique le prétraitement du texte (`preprocess_sent()`, `bag_of_words()`) et la similarité cosinus par rapport à `tfidfsymptoms.csv` pour identifier un symptôme probable à partir de l'entrée de texte libre de l'utilisateur.
    * Si plusieurs symptômes potentiels sont identifiés, le chatbot pose des questions de clarification "oui/non" pour confirmer le bon symptôme.

    .. code-block:: python

        # Extrait de app.py
        def predictSym(sym, vocab, app_tag):
            sym = preprocess_sent(sym)
            bow = np.array(bag_of_words(sym, vocab))
            # ... (calcul de la similarité cosinus) ...
            # ... (retourne les symptômes possibles et la confiance) ...

4.  **Collecte progressive des symptômes et affinement du diagnostic :**
    * Une fois les symptômes initiaux confirmés, le chatbot calcule les `possible_diseases()` en fonction des symptômes recueillis jusqu'à présent.
    * Si plusieurs maladies sont encore possibles, il pose de manière itérative des questions sur des symptômes supplémentaires associés à ces maladies candidates (`symVONdisease()`) que l'utilisateur n'a pas encore mentionnés. Cette interrogation interactive aide à affiner le diagnostic.

5.  **Prédiction des maladies :**
    * Lorsque suffisamment de symptômes sont collectés, ou si la liste des `possible_diseases` a été affinée (idéalement à une seule), le chatbot effectue une prédiction finale.
    * Les symptômes collectés sont convertis en un vecteur codé en un seul coup (OHV) à l'aide de la fonction `OHV()`, correspondant au format des données d'entraînement.
    * Le modèle KNN (`knn.predict()`) prédit ensuite la maladie la plus probable.

    .. code-block:: python

        # Extrait de app.py
        def make_final_prediction(chat_state):
            ohv_result = OHV(chat_state['symptoms'], all_symp_col)
            prediction = knn.predict(ohv_result)
            predicted_disease = prediction[0]
            # ... (stockage du diagnostic, mise à jour de l'état) ...

6.  **Évaluation de la gravité et précautions :**
    * Le chatbot demande à l'utilisateur la durée (nombre de jours) pendant laquelle il a ressenti les symptômes.
    * À l'aide de `calc_condition()`, il évalue la gravité en fonction de la somme des gravités des symptômes individuels (`severityDictionary`) et de la durée.
    * Il fournit ensuite des conseils généraux et, le cas échéant, une liste de précautions (`precautionDictionary`) spécifiques à la maladie diagnostiquée.

7.  **Réinitialisation de la conversation :**
    * À la fin d'une consultation, il est demandé à l'utilisateur s'il a besoin d'une autre consultation. Si "oui", le `chat_state` est réinitialisé, lançant une nouvelle conversation.

Résultats
=======
Le résultat principal du chatbot est un diagnostic conversationnel et des informations médicales pertinentes.

Interface du Chatbot :
------------------
L'interaction se déroule via une interface de chat web. Les résultats sont affichés directement dans la fenêtre de chat, fournissant un retour en temps réel à l'utilisateur.

.. figure:: images/GUI/chatbot_interface.png
   :alt: Interface utilisateur du Chatbot
   :width: 700px

   Une capture d'écran de l'interface web interactive, montrant un flux de conversation typique, la saisie des symptômes et la sortie du diagnostic.

Sorties attendues :
-----------------
* **Maladie prédite :** La maladie la plus probable en fonction des symptômes fournis.
* **Description de la maladie :** Une brève explication de la maladie prédite.
* **Évaluation de la gravité :** Une indication de la gravité de l'affection basée sur l'intensité et la durée des symptômes.
* **Précautions :** Actions ou précautions recommandées spécifiques à la maladie diagnostiquée.
* **Flux interactif :** La possibilité de clarifier les symptômes, de poser des questions sur des symptômes supplémentaires et de redémarrer la consultation.
