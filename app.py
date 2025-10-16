# app.py - Version avec ordre des colonnes corrigé
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="🍄 MushroomSafe AI",
    page_icon="🍄",
    layout="wide"
)

# Style CSS pour rendre joli
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .safe-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #155724;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 3px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 3px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


# Charger les modèles
@st.cache_resource
def load_models():
    try:
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

        return models, label_encoders, target_encoder, column_order, "✅ Modèles chargés avec succès!"

    except Exception as e:
        return None, None, None, None, f"❌ Erreur: {str(e)}"


def main():
    # Titre principal
    st.markdown('<h1 class="main-title">🍄 MushroomSafe AI</h1>', unsafe_allow_html=True)
    st.markdown("### 🔍 Votre Assistant Intelligent pour Identifier les Champignons")

    # Chargement des modèles
    models, label_encoders, target_encoder, column_order, status = load_models()

    # Sidebar
    st.sidebar.info(status)

    if models is None:
        st.error("❌ Impossible de charger les modèles. Avez-vous bien exécuté train_models.py ?")
        return

    # Avertissement de sécurité
    st.markdown("""
    <div class='info-box'>
    <h4>⚠️ ATTENTION IMPORTANTE</h4>
    <b>Cette application est à but éducatif uniquement.</b><br>
    Ne consommez jamais un champignon basé uniquement sur cette analyse !<br>
    Consultez toujours un expert en cas de doute.
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Saisie des caractéristiques
    st.header("📝 Décrivez Votre Champignon")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🍄 Chapeau")
        cap_shape = st.selectbox(
            "Forme du chapeau",
            ['x', 'b', 's', 'f', 'k', 'c'],
            format_func=lambda x: {
                'x': 'Convexe', 'b': 'En cloche', 's': 'Déprimé',
                'f': 'Plat', 'k': 'Bosse', 'c': 'Conique'
            }[x]
        )
        cap_color = st.selectbox(
            "Couleur du chapeau",
            ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
            format_func=lambda x: {
                'n': 'Marron', 'b': 'Beige', 'c': 'Cannelle', 'g': 'Gris',
                'r': 'Vert', 'p': 'Rose', 'u': 'Violet', 'e': 'Rouge',
                'w': 'Blanc', 'y': 'Jaune'
            }[x]
        )

    with col2:
        st.markdown("#### 👃 Odeur")
        odor = st.selectbox(
            "Odeur caractéristique",
            ['n', 'f', 'y', 's', 'a', 'l', 'p', 'c', 'm'],
            format_func=lambda x: {
                'n': 'Aucune', 'f': 'Fétide', 'y': 'Poisson', 's': 'Épicé',
                'a': 'Amande', 'l': 'Anis', 'p': 'Piquant', 'c': 'Crésote',
                'm': 'Moisi'
            }[x]
        )

    # Section 2: Autres caractéristiques
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🎨 Couleurs")
        gill_color = st.selectbox(
            "Couleur des lamelles",
            ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
            format_func=lambda x: {
                'k': 'Noir', 'n': 'Marron', 'b': 'Beige', 'h': 'Chocolat',
                'g': 'Gris', 'r': 'Vert', 'o': 'Orange', 'p': 'Rose',
                'u': 'Violet', 'e': 'Rouge', 'w': 'Blanc', 'y': 'Jaune'
            }[x]
        )

    with col4:
        st.markdown("#### 🌍 Environnement")
        habitat = st.selectbox(
            "Lieu de cueillette",
            ['g', 'l', 'm', 'p', 'u', 'w', 'd'],
            format_func=lambda x: {
                'g': 'Herbes', 'l': 'Feuilles', 'm': 'Prairies', 'p': 'Chemins',
                'u': 'Urbain', 'w': 'Déchets', 'd': 'Forêt'
            }[x]
        )

    # Section 3: Choix du modèle
    st.markdown("#### 🤖 Intelligence Artificielle")
    selected_model = st.selectbox(
        "Choisissez le modèle d'analyse :",
        list(models.keys())
    )

    # Bouton d'analyse
    if st.button("🔍 ANALYSER LA COMESTIBILITÉ", type="primary", use_container_width=True):
        with st.spinner(f'🔍 Analyse en cours avec {selected_model}...'):
            # Préparation des données
            input_data = prepare_input_data(cap_shape, cap_color, odor, gill_color, habitat, label_encoders,
                                            column_order)

            # Prédiction
            model = models[selected_model]
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            # Affichage du résultat
            display_prediction_result(prediction, probability, target_encoder, selected_model)


def prepare_input_data(cap_shape, cap_color, odor, gill_color, habitat, label_encoders, column_order):
    """Prépare les données pour le modèle avec le BON ordre de colonnes"""

    # Valeurs par défaut pour TOUTES les colonnes dans le BON ordre
    default_values = {
        'cap-shape': 'x',
        'cap-surface': 's',
        'cap-color': 'n',
        'bruises': 'f',
        'odor': 'n',
        'gill-attachment': 'f',
        'gill-spacing': 'c',
        'gill-size': 'b',
        'gill-color': 'k',
        'stalk-shape': 'e',
        'stalk-root': 'b',
        'stalk-surface-above-ring': 's',
        'stalk-surface-below-ring': 's',
        'stalk-color-above-ring': 'w',
        'stalk-color-below-ring': 'w',
        'veil-type': 'p',
        'veil-color': 'w',
        'ring-number': 'o',
        'ring-type': 'p',
        'spore-print-color': 'w',
        'population': 's',
        'habitat': 'g'
    }

    # Mettre à jour avec les valeurs de l'utilisateur
    user_values = {
        'cap-shape': cap_shape,
        'cap-color': cap_color,
        'odor': odor,
        'gill-color': gill_color,
        'habitat': habitat
    }

    default_values.update(user_values)

    # Créer le DataFrame dans le BON ordre
    input_data = []
    for col in column_order:
        input_data.append(default_values[col])

    input_df = pd.DataFrame([input_data], columns=column_order)

    # Encodage des variables
    for col in column_order:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    return input_df


def display_prediction_result(prediction, probability, target_encoder, model_name):
    """Affiche le résultat de l'analyse"""

    # Décodage de la prédiction
    prediction_label = target_encoder.inverse_transform([prediction])[0]
    confidence = probability[prediction] * 100

    st.markdown("---")
    st.header("🎯 RÉSULTAT DE L'ANALYSE")

    # Affichage du résultat principal
    if prediction_label == 'e':
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

    # Détails techniques
    with st.expander("📊 DÉTAILS TECHNIQUES"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Modèle utilisé", model_name)
        with col2:
            st.metric("Confiance", f"{confidence:.1f}%")
        with col3:
            prob_edible = probability[0] * 100
            prob_poisonous = probability[1] * 100
            st.metric("Probabilité comestible", f"{prob_edible:.1f}%")

    # Avertissement final
    st.warning("""
    **🔬 RAPPEL IMPORTANT :** 
    Cette analyse est réalisée par une intelligence artificielle et peut contenir des erreurs.
    Elle ne remplace pas l'expertise d'un mycologue professionnel.
    En cas de doute sur un champignon, ne le consommez pas et consultez un expert !
    """)


if __name__ == "__main__":
    main()
