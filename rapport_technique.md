# Rapport Technique — Détection d'Intrusions Réseau (DDoS) par Deep Learning

**Projet :** Network Intrusion Detection — CICDDoS2019  
**Domaine :** IA appliquée à la Cybersécurité  
**Problème :** Classification multi-classes supervisée — 7 classes  
**Environnement :** Google Colab (GPU T4)  
**Notebooks :** preprocessing · dnn\_model · cnn\_model · lstm\_model · cnn\_lstm\_model

---

## 1. Contexte et Motivation

Les attaques par déni de service distribué (DDoS) représentent aujourd'hui une menace critique pour les infrastructures réseau mondiales. Les systèmes de détection d'intrusions (IDS/NIDS) basés sur des signatures échouent face aux variantes nouvelles et aux attaques polymorphiques.

Le **deep learning** offre une approche complémentaire : plutôt que de comparer le trafic à des signatures connues, les modèles apprennent automatiquement des représentations statistiques du trafic bénin et malveillant, leur permettant de généraliser à de nouvelles attaques.

Ce projet compare quatre architectures — DNN, CNN, LSTM, CNN-LSTM — sur le même dataset et le même pipeline de données pour une évaluation rigoureuse.

---

## 2. Dataset : CICDDoS2019

### Présentation

Le **Canadian Institute for Cybersecurity DDoS Dataset 2019** est un benchmark de référence généré dans un environnement réseau réel sur deux jours de capture.

| Propriété | Valeur |
|-----------|--------|
| Source | University of New Brunswick (UNB), Canada |
| URL | https://www.unb.ca/cic/datasets/ddos-2019.html |
| Outil de capture | CICFlowMeter (features statistiques par flux) |
| Format brut | Fichiers CSV — train et test séparés |
| Features brutes | ~80 features statistiques réseau |

### Les 7 classes retenues

Après harmonisation des labels entre train et test :

| Label final | Label(s) source | Protocole |
|-------------|----------------|-----------|
| BENIGN | BENIGN | — |
| DrDoS_UDP | UDP | UDP |
| UDP-lag | UDPLag | UDP |
| DrDoS_MSSQL | MSSQL | UDP/1434 |
| DrDoS_LDAP | LDAP | UDP/389 |
| DrDoS_NetBIOS | NetBIOS | UDP/137 |
| Syn | Syn | TCP |

> Les attaques **DrDoS** (Distributed Reflection DoS) abusent de protocoles UDP : un attaquant envoie de petites requêtes en usurpant l'IP de la victime, et les serveurs tiers répondent avec des paquets bien plus volumineux, amplifiant ainsi le déni de service.

---

## 3. Pipeline de Prétraitement Complet

### 3.1 Alignement train / test

Le jeu de test contient des colonnes absentes du train (artefact de génération) :

```python
extra_cols_in_test = set(test_df.columns) - set(train_df.columns)
test_df.drop(columns=extra_cols_in_test, inplace=True)
```

Les noms de colonnes sont également strippés des espaces parasites.

### 3.2 Suppression des features inutiles (14 colonnes)

| Catégorie | Colonnes | Raison |
|-----------|----------|--------|
| TCP Flags peu informatifs | `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `FIN Flag Count`, `PSH Flag Count`, `ECE Flag Count` | Faible variance, faible contribution |
| Features Bulk redondantes | `Fwd/Bwd Avg Bytes/Bulk`, `Fwd/Bwd Avg Packets/Bulk`, `Fwd/Bwd Avg Bulk Rate` | Données redondantes et peu discriminantes |
| Colonne technique | `Unnamed: 0` | Index inutile |
| Donnée peu exploitable | `SimillarHTTP` | Trop de valeurs manquantes |

### 3.3 Nettoyage numérique

```python
# Infinis → NaN puis suppression des lignes concernées
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
```

### 3.4 Suppression des colonnes redondantes et dupliquées

- Suppression manuelle : `Fwd Header Length.1` (doublon exact de `Fwd Header Length`)
- Détection algorithmique : comparaison colonne à colonne (`df[col1].equals(df[col2])`) avec suppression des doublons détectés

### 3.5 Analyse et suppression des corrélations élevées

Suppression de toutes les features présentant une corrélation > 0.8 avec une autre feature :

```python
corr_matrix = numerical_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]
df.drop(columns=high_corr_cols, inplace=True)
```

> **Shape finale :** `(n_samples, 27)` — 27 features retenues après toutes les étapes de nettoyage.

### 3.6 Harmonisation et filtrage des labels

```python
label_mapping = {
    'UDP': 'DrDoS_UDP', 'UDPLag': 'UDP-lag',
    'MSSQL': 'DrDoS_MSSQL', 'LDAP': 'DrDoS_LDAP',
    'NetBIOS': 'DrDoS_NetBIOS', 'Syn': 'Syn', 'BENIGN': 'BENIGN'
}
test_df["Label"] = test_df["Label"].replace(label_mapping)
```

Filtrage sur les 7 classes retenues, reset des index.

### 3.7 Encodage, Normalisation, SMOTE

```python
# Encodage
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded  = label_encoder.transform(y_test)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)     # transform seulement → pas de data leakage

# Rééquilibrage des classes (train uniquement)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
```

> SMOTE (Synthetic Minority Over-sampling Technique) génère des échantillons synthétiques pour les classes minoritaires, uniquement sur le jeu d'entraînement pour éviter tout data leakage vers le test.

### 3.8 Sauvegarde des artefacts

```
X_train.npy, y_train.npy  ← données après SMOTE
X_test.npy,  y_test.npy   ← données normalisées uniquement
scaler.pkl, label_encoder.pkl ← sauvegardés avec joblib
```

### 3.9 Split interne à chaque notebook modèle

Un second split stratifié est appliqué dans chaque notebook :

```python
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
```

**Répartition effective finale :** 64% train · 16% validation · 20% test

---

## 4. Architectures des Modèles

### 4.1 DNN — Deep Neural Network

Modèle de référence (baseline) entièrement connecté.

```
Input(27)
  Dense(256, relu) + BatchNorm + Dropout(0.2)
  Dense(128, relu) + BatchNorm + Dropout(0.2)
  Dense(64,  relu) + BatchNorm + Dropout(0.2)
  Dense(32,  relu) + Dropout(0.2)
  Dense(7, softmax)
```

| Paramètre | Valeur |
|-----------|--------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 128 |
| Epochs max | 20 |
| EarlyStopping patience | 7 |
| ReduceLROnPlateau | factor=0.5, patience=3 |

Seul modèle à utiliser `ReduceLROnPlateau` — choix justifié par la profondeur du réseau (5 couches denses).

---

### 4.2 CNN — Convolutional Neural Network 1D

Le vecteur de 27 features est traité comme une séquence 1D `(27, 1)`.

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

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Epochs max | 30 |
| EarlyStopping patience | 5 |

Architecture plus légère que le DNN, mais 3 couches convolutives en cascade pour une extraction hiérarchique de features.

---

### 4.3 LSTM — Long Short-Term Memory

Le vecteur de features est interprété comme une séquence temporelle `(27, 1)`.

```
Input(27, 1)
  LSTM(64, return_sequences=True) + Dropout(0.2)
  LSTM(32) + Dropout(0.2)
  Dense(32, relu)
  Dense(7, softmax)
```

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Epochs max | 20 |
| EarlyStopping patience | 5 |

Architecture symétrique au CNN-LSTM sans la partie convolutive — permet d'évaluer la contribution spécifique du LSTM.

---

### 4.4 CNN-LSTM — Architecture Hybride ⭐

Combine l'extraction locale des convolutions et la mémoire séquentielle du LSTM.

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

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Epochs max | 30 |
| EarlyStopping patience | 5 |

Différences clés vs CNN : `padding=same` sur les convolutions (préserve la dimension spatiale pour le LSTM suivant), BatchNorm entre les couches Conv et LSTM.

---

## 5. Stratégie d'Entraînement Commune

### Compilation (tous modèles)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks communs

| Callback | Config commune | Spécificité |
|----------|---------------|-------------|
| `EarlyStopping` | `monitor='val_loss'`, `restore_best_weights=True` | `patience=7` pour DNN, `patience=5` pour les autres |
| `ModelCheckpoint` | `monitor='val_accuracy'`, `save_best_only=True`, `mode='max'` | — |
| `ReduceLROnPlateau` | `factor=0.5, patience=3` | DNN uniquement |

---

## 6. Évaluation

### Métriques calculées pour chaque modèle

- Accuracy et loss sur le jeu de test
- `classification_report` : precision, recall, F1-score par classe + weighted avg
- Matrice de confusion (heatmap Seaborn, `cmap='Blues'`)
- Courbes accuracy train/validation par epoch

### Artefacts sauvegardés par modèle

| Fichier | Contenu |
|---------|---------|
| `models/best_*_model_7_classes.h5` | Meilleurs poids (val_accuracy max) |
| `metrics/*_metrics.pkl` | `{'accuracy': float, 'loss': float}` |
| `history/*_history.pkl` | Dict `history.history` complet |
| `predictions/*_pred.npy` | Prédictions entières `y_pred` sur X_test |
| `predictions/y_test.npy` | Labels réels (référence commune à tous les modèles) |

---

## 7. Tableau Comparatif des Architectures

| Critère | DNN | CNN | LSTM | CNN-LSTM |
|---------|-----|-----|------|----------|
| Input shape | `(27,)` | `(27, 1)` | `(27, 1)` | `(27, 1)` |
| Paramètres approx. | ~90K | ~15K | ~45K | ~60K |
| BatchNorm | ✅ (3 couches) | ❌ | ❌ | ✅ (2 couches) |
| Dropout | 0.2 × 4 | 0.2 × 1 | 0.2 × 2 | 0.2 × 2 |
| Epochs max | 20 | 30 | 20 | 30 |
| Batch size | 128 | 64 | 64 | 64 |
| ReduceLR | ✅ | ❌ | ❌ | ❌ |
| Complexité | Moyenne | Faible | Moyenne | Élevée |

---

## 8. Limitations et Perspectives

### Limitations

- Dataset statique — pas de généralisation testée sur d'autres captures réseau
- Certaines classes DDoS ont des signatures proches (UDP vs UDP-lag)
- L'évaluation n'inclut pas de test de robustesse aux données adversariales

### Perspectives

- **Temps réel :** intégration avec Scapy ou Zeek pour classer des flux live
- **Non-supervisé :** AutoEncoder pour détecter des attaques inconnues (zero-day)
- **Explicabilité :** SHAP / LIME pour identifier les features les plus discriminantes
- **Déploiement :** API REST (FastAPI) + containerisation Docker
- **Données :** extension à CICDDoS2020 ou UNSW-NB15

---

## 9. Références

1. Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A.A. (2019). *Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy.* ICCST 2019.
2. [CICDDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) — Canadian Institute for Cybersecurity.
3. Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
4. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.

---

*Rapport généré dans le cadre d'un portfolio Ingénieur IA / Cybersécurité*
