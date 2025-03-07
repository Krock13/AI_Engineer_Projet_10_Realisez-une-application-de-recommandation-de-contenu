import streamlit as st
import requests
import json

# Configuration de l'URL de l'Azure Function
AZURE_FUNCTION_URL = "https://recommender-function.azurewebsites.net/api/recommend"

# Interface Streamlit
st.title("Recommandation d'Articles")
st.markdown("Entrez un identifiant utilisateur pour obtenir des recommandations personnalisées.")

# Saisie de l'ID utilisateur
user_id = st.text_input("ID Utilisateur :", "")

def is_valid_user_id(user_id):
    """Vérifie si l'ID utilisateur est un entier valide entre 0 et 322896."""
    try:
        user_id = int(user_id)
        return 0 <= user_id <= 322896
    except ValueError:
        return False

if st.button("Obtenir des recommandations"):
    if not is_valid_user_id(user_id):
        st.error("L'ID utilisateur doit être un nombre entre 0 et 322896.")
    else:
        try:
            response = requests.get(f"{AZURE_FUNCTION_URL}?user_id={user_id}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.success(f"Recommandations pour l'utilisateur {data['user_id']} :")

                    for idx, article in enumerate(data["recommendations"], 1):
                        st.write(f"{idx}. Article ID: {article}")
                except json.JSONDecodeError:
                    st.error("Erreur : Réponse invalide du serveur. Veuillez réessayer.")
            elif response.status_code == 404:
                st.warning("Cet utilisateur est inconnu. Veuillez vérifier l'ID saisi ou essayer avec un autre utilisateur.")
            elif response.status_code == 500:
                st.error("Le service de recommandation est temporairement surchargé. Veuillez réessayer plus tard.")
            else:
                st.error(f"Erreur {response.status_code} : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {str(e)}")

# Ajout d'un style pour rendre l'affichage plus agréable
st.markdown(
    """
    <style>
    .stButton>button {
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
