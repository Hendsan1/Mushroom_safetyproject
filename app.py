# app.py - Version auto-création des modèles
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration
st.set_page_config(
    page_title="🍄 MushroomSafe AI",
    page_icon="🍄",
    layout="wide"
)

# Style CSS
st.markdown("""
<style>
    .main-title { text-align: center; color: #2E8B57; font-size: 3rem; margin-bottom: 1rem; }
    .safe-box { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); color: #155724; padding: 2rem; border-radius: 15px; text-align: center; border: 3px solid #28a745; margin: 1rem 0; }
    .danger-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; border: 3px solid #dc3545; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_create_models():
    """Charge les modèles ou les crée si nécessaire"""
    try:
        # Essayer de charger les modèles existants
        if os.path.exists('models/random_forest.pkl'):
            models = {}
            model_files = {
                'Random Forest': 'random_forest.pkl',
                'Logistic Regression': 'logistic_regression.pkl',
                'Neural Network': 'neural_network.pkl', 
                'Decision Tree': 'decision_tree.pkl'
            }
            
            for name, file in model_files.items():
                models[name] = joblib.load(f'models/{file}')
            
            label_encoders = joblib.load('models/label_encoders.pkl')
            target_encoder = joblib.load('models/target_encoder.pkl')
            column_order = joblib.load('models/column_order.pkl')
            
            return models, label_encoders, target_encoder, column_order, "✅ Modèles chargés!"
        else:
            # Créer les modèles
            return create_models()
    except Exception as e:
        # Créer les modèles en cas d'erreur
        return create_models()

def create_models():
    """Crée les modèles de ML"""
    st.info("🔄 Création des modèles IA... (premier lancement)")
    
    # Créer le dossier models
    os.makedirs('models', exist_ok=True)
    
    # Charger les données
    df = pd.read_csv('mushrooms.csv')
    
    # Préparer les données
    label_encoders = {}
    X_encoded = pd.DataFrame()
    
    column_order = [col for col in df.columns if col != 'class']
    
    for col in column_order:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le
        X_encoded[col] = le.transform(df[col])
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df['class'])
    
    # Entraîner les modèles
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50,)),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_encoded, y_encoded)
    
    # Sauvegarder les modèles
    for name, model in models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
    
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(le_target, 'models/target_encoder.pkl')
    joblib.dump(column_order, 'models/column_order.pkl')
    
    return models, label_encoders, le_target, column_order, "✅ Modèles créés avec succès!"

def main():
    st.markdown('<h1 class="main-title">🍄 MushroomSafe AI</h1>', unsafe_allow_html=True)
    st.markdown("### 🔍 Votre Assistant Intelligent pour Identifier les Champignons")
    
    # Charger/créer les modèles
    models, label_encoders, target_encoder, column_order, status = load_or_create_models()
    st.sidebar.info(status)
    
    # Interface utilisateur
    st.header("📝 Décrivez Votre Champignon")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cap_shape = st.selectbox("Forme du chapeau", ['x', 'b', 's', 'f', 'k', 'c'],
                               format_func=lambda x: {'x': 'Convexe', 'b': 'En cloche', 's': 'Déprimé', 'f': 'Plat', 'k': 'Bosse', 'c': 'Conique'}[x])
        cap_color = st.selectbox("Couleur du chapeau", ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
                               format_func=lambda x: {'n': 'Marron', 'b': 'Beige', 'c': 'Cannelle', 'g': 'Gris', 'r': 'Vert', 'p': 'Rose', 'u': 'Violet', 'e': 'Rouge', 'w': 'Blanc', 'y': 'Jaune'}[x])
    
    with col2:
        odor = st.selectbox("Odeur", ['n', 'f', 'y', 's', 'a', 'l', 'p', 'c', 'm'],
                          format_func=lambda x: {'n': 'Aucune', 'f': 'Fétide', 'y': 'Poisson', 's': 'Épicé', 'a': 'Amande', 'l': 'Anis', 'p': 'Piquant', 'c': 'Crésote', 'm': 'Moisi'}[x])
        gill_color = st.selectbox("Couleur des lamelles", ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
                                format_func=lambda x: {'k': 'Noir', 'n': 'Marron', 'b': 'Beige', 'h': 'Chocolat', 'g': 'Gris', 'r': 'Vert', 'o': 'Orange', 'p': 'Rose', 'u': 'Violet', 'e': 'Rouge', 'w': 'Blanc', 'y': 'Jaune'}[x])
    
    habitat = st.selectbox("Habitat", ['g', 'l', 'm', 'p', 'u', 'w', 'd'],
                         format_func=lambda x: {'g': 'Herbes', 'l': 'Feuilles', 'm': 'Prairies', 'p': 'Chemins', 'u': 'Urbain', 'w': 'Déchets', 'd': 'Forêt'}[x])
    
    selected_model = st.selectbox("Modèle IA", list(models.keys()))
    
    if st.button("🔍 ANALYSER LA COMESTIBILITÉ", type="primary", use_container_width=True):
        # Préparation des données
        input_dict = {
            'cap-shape': cap_shape, 'cap-surface': 's', 'cap-color': cap_color,
            'bruises': 'f', 'odor': odor, 'gill-attachment': 'f',
            'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': gill_color,
            'stalk-shape': 'e', 'stalk-root': 'b', 'stalk-surface-above-ring': 's',
            'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'w',
            'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w',
            'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'w',
            'population': 's', 'habitat': habitat
        }
        
        input_data = []
        for col in column_order:
            input_data.append(input_dict[col])
        
        input_df = pd.DataFrame([input_data], columns=column_order)
        
        # Encodage
        for col in column_order:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Prédiction
        model = models[selected_model]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Affichage résultat
        result = target_encoder.inverse_transform([prediction])[0]
        confidence = probability[prediction] * 100
        
        st.markdown("---")
        st.header("🎯 RÉSULTAT DE L'ANALYSE")
        
        if result == 'e':
            st.markdown(f"""
            <div class='safe-box'>
            <h1>✅ CHAMPIGNON COMESTIBLE</h1>
            <p style='font-size: 1.3rem;'>Selon l'analyse IA, ce champignon est probablement sans danger</p>
            <p style='font-size: 1.1rem;'>Niveau de confiance: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class='danger-box'>
            <h1>☠️ CHAMPIGNON POISONNEUX</h1>
            <p style='font-size: 1.3rem;'>⚠️ DANGER - Ne pas consommer !</p>
            <p style='font-size: 1.1rem;'>Niveau de confiance: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.warning("⚠️ Application éducative - Ne pas utiliser pour la consommation réelle")

if __name__ == "__main__":
    main()
