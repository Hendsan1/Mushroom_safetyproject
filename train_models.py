# train_models.py - Version avec ordre des colonnes fixe
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("🍄 Début de l'entraînement des modèles...")

# 1. Chargement des données
df = pd.read_csv('mushrooms.csv')
print(f"✅ Données chargées: {df.shape[0]} champignons, {df.shape[1]} caractéristiques")

# 2. Afficher l'ordre des colonnes
print(f"📋 Ordre des colonnes: {list(df.columns)}")

# 3. Préparation des données
print("\n🔧 Préparation des données...")

label_encoders = {}
X_encoded = pd.DataFrame()

# IMPORTANT: Garder le même ordre de colonnes que le dataset original
column_order = list(df.columns)
column_order.remove('class')  # Retirer la cible

# Utiliser les VRAIES valeurs du dataset pour l'encodage
for col in column_order:
    le = LabelEncoder()
    # Utiliser les valeurs réelles du dataset
    le.fit(df[col])
    label_encoders[col] = le
    X_encoded[col] = le.transform(df[col])

# S'assurer que l'ordre des colonnes est correct
X_encoded = X_encoded[column_order]

# Encodage de la cible
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df['class'])

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Données préparées - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"📊 Ordre final des colonnes: {list(X_encoded.columns)}")

# 4. Définition des 4 modèles
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100,)),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# 5. Entraînement des modèles
print("\n🚀 Entraînement des 4 modèles...")
results = {}

for name, model in models.items():
    print(f"🔧 Entraînement de {name}...")
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': accuracy
    }

    print(f"   ✅ {name} - Accuracy: {accuracy:.4f}")

# 6. Sauvegarde des modèles
print("\n💾 Sauvegarde des modèles...")

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Sauvegarder chaque modèle
for name, result in results.items():
    filename = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(result['model'], filename)
    print(f"   ✅ {filename} sauvegardé")

# Sauvegarder les encodeurs
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(le_target, 'models/target_encoder.pkl')

# Sauvegarder l'ordre des colonnes
joblib.dump(column_order, 'models/column_order.pkl')

print("\n🎉 ENTRAÎNEMENT TERMINÉ !")
print("📊 Résultats des modèles :")
for name, result in results.items():
    print(f"   • {name}: {result['accuracy']:.1%}")

