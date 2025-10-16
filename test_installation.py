# test_installation.py - Vérifie que tout est installé
print("🔍 Vérification des installations...")

try:
    import pandas as pd
    print("✅ pandas installé")
except:
    print("❌ pandas NON installé")

try:
    import sklearn
    print("✅ scikit-learn installé")
except:
    print("❌ scikit-learn NON installé")

try:
    import joblib
    print("✅ joblib installé")
except:
    print("❌ joblib NON installé")

try:
    import streamlit
    print("✅ streamlit installé")
except:
    print("❌ streamlit NON installé")

try:
    import plotly
    print("✅ plotly installé")
except:
    print("❌ plotly NON installé")

print("\n🎯 Test terminé!")
