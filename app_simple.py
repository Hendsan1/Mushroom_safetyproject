# app_simple.py - Version de test
import streamlit as st

st.set_page_config(page_title="🍄 MushroomSafe AI", page_icon="🍄")

st.title("🍄 MushroomSafe AI - TEST")
st.success("✅ Streamlit Share FONCTIONNE !")

st.write("Cette version simple prouve que le déploiement marche.")
st.write("Prochaine étape : ajouter l'IA complète")

if st.button("🎯 Test réussi"):
    st.balloons()
    st.info("🚀 Prêt pour la version IA complète !")
