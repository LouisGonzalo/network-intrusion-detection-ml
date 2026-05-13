# 🔐 Comparative Network Intrusion Detection using Machine Learning and Deep Learning

## 📌 Description

Ce projet présente une approche comparative de détection d’intrusion réseau basée sur des techniques de Machine Learning et de Deep Learning.

L’objectif principal est d’identifier automatiquement les comportements anormaux ou malveillants dans le trafic réseau à travers plusieurs modèles d’apprentissage afin d’évaluer leurs performances respectives.

Ce travail a été réalisé dans le cadre de mon mémoire de fin d’études en sécurité informatique.

---

## 👤 Auteur

**Louis Kodjo ADETI**  
Ingénieur en sécurité informatique  
Togo

---

## 🎓 Formation et certification

- Licence fondamentale en Mathématiques
- Diplôme d’ingénieur en sécurité informatique
- Cisco Networking Academy – Introduction to Cybersecurity

---

## 🎯 Objectifs du projet

- Détecter les intrusions réseau
- Classifier différents types d’attaques
- Comparer plusieurs modèles d’apprentissage
- Évaluer les performances des modèles
- Identifier les approches les plus efficaces pour la détection d’anomalies réseau

---

# 🧠 Modèles utilisés

## 🔹 Machine Learning
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Logistic Regression
- Naive Bayes

## 🔹 Deep Learning
- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- CNN-LSTM Hybride

---

# ⚙️ Pipeline du projet

1. Collecte et préparation des données réseau
2. Nettoyage des données
3. Prétraitement et normalisation
4. Encodage des labels
5. Séparation des données d’entraînement et de test
6. Entraînement des modèles ML/DL
7. Évaluation des performances
8. Analyse comparative des résultats

---

# 📂 Structure du projet

```text
network-intrusion-detection-ml/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── machine_learning.ipynb
│   └── deep_learning.ipynb
│
├── models/
│   ├── machine_learning/
│   └── deep_learning/
│
├── metrics/
├── history/
├── predictions/
│
├── results/
│   ├── confusion_matrices/
│   ├── reports/
│   ├── training_curves/
│   └── comparison/
│
└── data/
```

---

# 📊 Résultats

## 🔹 Matrices de confusion

### DNN
![DNN Confusion Matrix](results/confusion_matrices/matrice_confusion_dnn.png)

### LSTM
![LSTM Confusion Matrix](results/confusion_matrices/matrice_confusion_lstm.png)

### Random Forest
![Random Forest Confusion Matrix](results/confusion_matrices/RandomForest_confusion.png)

### KNN
![KNN Confusion Matrix](results/confusion_matrices/KNN_confusion.png)

---

# 📈 Courbes d’entraînement

### CNN
![CNN Training](results/training_curves/courbes_train_val_test_cnn.png)

### LSTM
![LSTM Training](results/training_curves/courbes_train_val_test_lstm.png)

### DNN
![DNN Accuracy/Loss](results/training_curves/courbes_accuracy_loss_dnn.png)

---

# 📊 Comparaison des modèles

Le projet inclut une analyse comparative des performances des modèles de Machine Learning et de Deep Learning à travers plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score

### Résumé comparatif
![Model Comparison](results/comparison/resume_comparatif.png)

---

# 🛠️ Technologies utilisées

## 🔹 Langages et bibliothèques
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## 🔹 Outils
- Jupyter Notebook
- Wireshark
- Git / GitHub

---

# 🔍 Fonctionnalités principales

- Prétraitement automatisé des données
- Normalisation des caractéristiques
- Encodage des labels
- Entraînement de plusieurs modèles
- Évaluation comparative
- Visualisation des performances
- Analyse des résultats

---

# 🚀 Perspectives

- Détection d’intrusion en temps réel
- Déploiement cloud
- Intégration SIEM
- Optimisation des modèles
- Détection avancée basée sur le Deep Learning

---

# 🔐 Domaine

- Cybersécurité
- Sécurité réseau
- Intrusion Detection Systems (IDS)
- Machine Learning
- Deep Learning
- Analyse de trafic réseau

---

# 📌 Conclusion

Ce projet met en œuvre plusieurs techniques de Machine Learning et de Deep Learning appliquées à la cybersécurité afin d’évaluer leur capacité à détecter efficacement les intrusions réseau.

Les résultats obtenus montrent l’intérêt des approches intelligentes dans l’amélioration des systèmes de détection d’intrusion modernes.
