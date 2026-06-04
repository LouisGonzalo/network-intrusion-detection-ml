# 🛡️ Network Intrusion Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-yellow)
![Dataset](https://img.shields.io/badge/Dataset-CICDDoS2019-purple)
![Classes](https://img.shields.io/badge/Classes-7-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Classification multi-classes du trafic réseau (BENIGN + 6 types d'attaques DDoS) à l'aide de 4 architectures Deep Learning — **DNN, CNN, LSTM, CNN-LSTM** — sur le dataset **CICDDoS2019**. Inclut un pipeline complet de prétraitement avec SMOTE pour le rééquilibrage des classes.
>
> Projet réalisé dans le cadre d'un portfolio **Ingénieur IA / Cybersécurité**.

---

## 📌 Présentation

Ce projet implémente un **système de détection d'intrusions réseau (NIDS)** basé sur le deep learning. À partir de features statistiques extraites de flux réseau (dataset CICDDoS2019), quatre architectures sont entraînées, comparées et sauvegardées. Le pipeline comprend un prétraitement rigoureux incluant la suppression des corrélations élevées, la normalisation et le rééquilibrage des classes par **SMOTE**.

---

## 🎯 Objectifs

- ✅ Construire un pipeline de prétraitement complet et reproductible
- ✅ Classifier le trafic en **7 classes** (BENIGN + 6 variantes DDoS)
- ✅ Entraîner et comparer **4 architectures Deep Learning** sur les mêmes données
- ✅ Évaluer via accuracy, F1-score, matrices de confusion et courbes d'apprentissage

---

## 🗂️ Structure du projet

```
network-intrusion-detection-ml/
│
├── 📓 notebooks/
│   ├── preprocessing.ipynb             # Pipeline complet de prétraitement
│   ├── dnn_model.ipynb                 # Deep Neural Network
│   ├── cnn_model.ipynb                 # Convolutional Neural Network 1D
│   ├── lstm_model.ipynb                # Long Short-Term Memory
│   └── cnn_lstm_model.ipynb            # Hybride CNN-LSTM
│
├── 🤖 models/
│   ├── best_dnn_model_7_classes.h5
│   ├── best_cnn_model_7_classes.h5
│   ├── best_lstm_model_7_classes.h5
│   ├── best_cnn_lstm_model_7_classes.h5
│   └── label_encoder.pkl        # Sauvegardé avec joblib
    └── scaler.pkl               
│
├── 📊 metrics/                         # {'accuracy': float, 'loss': float}
│   ├── DNN_metrics.pkl
│   ├── CNN_metrics.pkl
│   ├── LSTM_metrics.pkl
│   └── CNN_LSTM_metrics.pkl
│
├── 📈 history/                         # history.history dict complet
│   ├── DNN_history.pkl
│   ├── CNN_history.pkl
│   ├── LSTM_history.pkl
│   └── CNN_LSTM_history.pkl
│
├── 🔮 predictions/
│   ├── DNN_pred.npy
│   ├── CNN_pred.npy
│   ├── LSTM_pred.npy
│   ├── CNN_LSTM_pred.npy
│   └── y_test.npy                      # Labels réels (référence commune)
│
├── 🖼️ results/
│   ├── courbes_accuracy_loss_dnn.png
│   ├── courbes_train_val_test_cnn.png
│   ├── courbes_train_val_test_lstm.png
│   ├── matrice_confusion_dnn.png
│   └── matrice_confusion_lstm.png
│
├── 📁 data/                            # ⚠️ Non versionné (voir .gitignore)
│   ├── X_train.npy                     # Données après SMOTE
│   ├── y_train.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   ├── scaler.pkl                      # StandardScaler sauvegardé
│   └── label_encoder.pkl
│
└── 📄 docs/
    ├── rapport_technique.md
    └── guide_github_debutant.md
```

---

## 🔄 Pipeline de Prétraitement

Le notebook `preprocessing.ipynb` réalise l'intégralité du pipeline avant l'entraînement.

```
CSV brut (CICDDoS2019)
    │
    ├─ 1. Alignement train/test (suppression colonnes extra du test)
    ├─ 2. Suppression de 14 features inutiles (flags TCP, bulk, index)
    ├─ 3. Remplacement inf/-inf → NaN puis dropna()
    ├─ 4. Suppression des doublons
    ├─ 5. Suppression colonne redondante (Fwd Header Length.1)
    ├─ 6. Détection et suppression des colonnes dupliquées (contenu identique)
    ├─ 7. Suppression des features corrélées à > 0.8 (matrice de corrélation)
    ├─ 8. Harmonisation des labels (UDP→DrDoS_UDP, MSSQL→DrDoS_MSSQL, etc.)
    ├─ 9. Filtrage sur les 7 classes retenues
    ├─ 10. LabelEncoder → fit sur train, transform sur test
    ├─ 11. StandardScaler → fit sur train, transform sur test
    ├─ 12. SMOTE sur X_train uniquement (éviter data leakage)
    │
    └─ Sauvegarde : X_train.npy, y_train.npy, X_test.npy, y_test.npy,
                    scaler.pkl, label_encoder.pkl
```

> **Input shape final :** `(n_samples, 27)` — 27 features après nettoyage

---

## 📊 Dataset et Classes

| Propriété | Valeur |
|-----------|--------|
| Dataset | CICDDoS2019 — Canadian Institute for Cybersecurity |
| URL | https://www.unb.ca/cic/datasets/ddos-2019.html |
| Features finales | 27 (après prétraitement) |
| Classes | 7 |

### Les 7 classes retenues

| Label | Type d'attaque | Protocole |
|-------|---------------|-----------|
| **BENIGN** | Trafic normal | — |
| **DrDoS_UDP** | Amplification UDP | UDP |
| **UDP-lag** | UDP avec délai (lag) | UDP |
| **DrDoS_MSSQL** | Amplification MSSQL | UDP/1434 |
| **DrDoS_LDAP** | Amplification LDAP | UDP/389 |
| **DrDoS_NetBIOS** | Amplification NetBIOS | UDP/137 |
| **Syn** | SYN Flood | TCP |

---

## 🧠 Architectures des 4 modèles

### DNN — Deep Neural Network

```
Input(27)
  Dense(256, relu) + BatchNorm + Dropout(0.2)
  Dense(128, relu) + BatchNorm + Dropout(0.2)
  Dense(64,  relu) + BatchNorm + Dropout(0.2)
  Dense(32,  relu) + Dropout(0.2)
  Dense(7, softmax)
```
Callbacks : `EarlyStopping(patience=7)` + `ModelCheckpoint` + `ReduceLROnPlateau(factor=0.5, patience=3)`  
Entraînement : 20 epochs max, batch=128

---

### CNN — Convolutional Neural Network 1D

```
Input(27, 1)
  Conv1D(16,  kernel=3, relu)
  Conv1D(32,  kernel=3, relu)
  Conv1D(64,  kernel=3, relu)
  MaxPooling1D(2)
  Flatten
  Dense(64, relu) + Dropout(0.2)
  Dense(7, softmax)
```
Callbacks : `EarlyStopping(patience=5)` + `ModelCheckpoint`  
Entraînement : 30 epochs max, batch=64

---

### LSTM — Long Short-Term Memory

```
Input(27, 1)
  LSTM(64, return_sequences=True) + Dropout(0.2)
  LSTM(32) + Dropout(0.2)
  Dense(32, relu)
  Dense(7, softmax)
```
Callbacks : `EarlyStopping(patience=5)` + `ModelCheckpoint`  
Entraînement : 20 epochs max, batch=64

---

### CNN-LSTM — Architecture Hybride ⭐

```
Input(27, 1)
  Conv1D(32, kernel=3, relu, padding=same) + BatchNorm
  Conv1D(64, kernel=3, relu, padding=same) + BatchNorm
  MaxPooling1D(2)
  LSTM(64, return_sequences=True) + Dropout(0.2)
  LSTM(32) + Dropout(0.2)
  Dense(32, relu)
  Dense(7, softmax)
```
Callbacks : `EarlyStopping(patience=5)` + `ModelCheckpoint`  
Entraînement : 30 epochs max, batch=64

---

## ⚙️ Installation

```bash
git clone https://github.com/<TON_USERNAME>/network-intrusion-detection-ml.git
cd network-intrusion-detection-ml

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Ordre d'exécution recommandé

```
1. notebooks/preprocessing.ipynb      → génère les .npy et .pkl
2. notebooks/dnn_model.ipynb
3. notebooks/cnn_model.ipynb
4. notebooks/lstm_model.ipynb
5. notebooks/cnn_lstm_model.ipynb
```

### Inférence sur un modèle sauvegardé

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Charger le modèle souhaité
model = load_model('models/best_cnn_lstm_model_7_classes.h5')
label_encoder = joblib.load('models/label_encoder.pkl')

# Charger et reshaper les données de test
X_test = np.load('data/X_test.npy')
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Prédiction
y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
print(label_encoder.inverse_transform(y_pred[:10]))
```

### Comparer les 4 modèles

```python
import pickle

for name in ['DNN', 'CNN', 'LSTM', 'CNN_LSTM']:
    with open(f'metrics/{name}_metrics.pkl', 'rb') as f:
        m = pickle.load(f)
    print(f"{name:10s} | Accuracy: {m['accuracy']:.4f} | Loss: {m['loss']:.4f}")
```

---

## 📈 Résultats

### Courbes d'apprentissage

| CNN | LSTM | DNN | CNN-LSTM |
|-----|------|-----|
| ![DNN](Results/figures/dnn_training_curves.png) | ![CNN](Results/figures/cnn_training_curves.png) | ![LSTM](Results/figures/lstm_training_curves.png) | ![CNN-LSTM](Results/figures/cnn_lstm_training_curves.png) |

### Matrices de confusion

| DNN | CNN | LSTM | CNN-LSTM |
|-----|------|
| ![DNN](Results/figures/DNN_confusion_matrix.png) | ![CNN](Results/figures/CNN_confusion_matrix.png) | ![LSTM](Results/figures/LSTM_confusion_matrix.png) | ![CNN-LSTM](Results/figures/CNN_LSTM_confusion_matrix.png) |

---

## 🛠️ Stack technique

| Outil | Usage |
|-------|-------|
| TensorFlow / Keras | Modèles DNN, CNN, LSTM, CNN-LSTM |
| Scikit-learn | LabelEncoder, StandardScaler, train_test_split, métriques |
| imbalanced-learn | SMOTE pour rééquilibrage des classes |
| NumPy | Arrays, sauvegarde `.npy` |
| Pandas | Chargement et nettoyage du dataset CSV |
| Matplotlib / Seaborn | Courbes, matrices de confusion, heatmaps |
| Joblib / Pickle | Sérialisation scaler, encoder, history, metrics |
| Google Colab | Environnement d'entraînement (GPU T4) |

---

## 👤 Auteur

**Louis Kodjo ADETI**  
🎓 Ingénieur en Sécurité Informatique | Network Security | Linux | Administration Système | Détection d'Intrusion Réseau | Ouvert aux Opportunités

- 🔗 [LinkedIn](https://www.linkedin.com/in/louis-adeti-b43018321)
- 💻 [GitHub](https://github.com/LouisGonzalo)
- 📧 [louisadeti18@gmail.com](mailto:louisadeti18@gmail.com)

---

## 📄 Licence

Ce projet est sous licence MIT — voir [LICENSE](LICENSE).

---

> 🎓 *Projet réalisé dans le cadre d'un portfolio Ingénieur IA / Cybersécurité*
