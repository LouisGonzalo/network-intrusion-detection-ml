# 🔐 Network Intrusion Detection using Machine Learning

## 📌 Description

Ce projet présente une approche de détection d’intrusion réseau basée sur le machine learning.

L’objectif est d’identifier automatiquement les comportements anormaux ou malveillants dans le trafic réseau à partir de plusieurs modèles d’apprentissage profond.

Ce travail a été réalisé dans le cadre de mon mémoire de fin d’études en sécurité informatique.

---

## 👤 Auteur

Louis Kodjo ADETI  
Ingénieur en sécurité informatique  
Togo

---

## 🎯 Objectifs du projet

- Détecter les intrusions réseau
- Classifier différents types d’attaques
- Comparer plusieurs modèles de Deep Learning
- Évaluer les performances des modèles

---

## 🧠 Modèles utilisés

- DNN (Deep Neural Network)
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- CNN-LSTM Hybride

---

## ⚙️ Pipeline du projet

1. Collecte et préparation des données
2. Prétraitement et normalisation
3. Encodage des labels
4. Entraînement des modèles
5. Évaluation des performances
6. Analyse des résultats

---

## 📂 Structure du projet

```text
network-intrusion-detection-ml/
│
├── notebooks/
├── models/
├── metrics/
├── history/
├── predictions/
├── results/
└── data/
```

---

## 📊 Résultats

### 🔹 Matrices de confusion

#### DNN
![DNN Confusion Matrix](results/matrice_confusion_dnn.png)

#### LSTM
![LSTM Confusion Matrix](results/matrice_confusion_lstm.png)

---

## 📈 Courbes d’entraînement

### CNN
![CNN Training](results/courbes_train_val_test_cnn.png)

### LSTM
![LSTM Training](results/courbes_train_val_test_lstm.png)

---

## 🛠️ Technologies utilisées

### Langages et bibliothèques
- Python
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib

### Outils
- Jupyter Notebook
- Wireshark
- Git / GitHub

---

## 📂 Dataset et modèles

Le dataset complet ainsi que les modèles entraînés (.h5) sont disponibles via Google Drive.

🔗 Ajouter ici le lien Google Drive

---

## 🚀 Perspectives

- Détection d’intrusion en temps réel
- Intégration avec SIEM
- Optimisation des performances
- Déploiement cloud

---

## 🔐 Domaine

- Cybersécurité
- Sécurité réseau
- Machine Learning
- Deep Learning
- Intrusion Detection Systems (IDS)
