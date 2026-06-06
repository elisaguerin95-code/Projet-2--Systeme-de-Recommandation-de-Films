import streamlit as st
import pandas as pd
import joblib 
import numpy as np
# pour la musique Netflix
import base64
# pour traduire en français le résumé
from deep_translator import GoogleTranslator
# l'authentificateur
from streamlit_authenticator import Authenticate
# le bandeau hero Wildflix
from bandeau import afficher_bandeau
import requests
import io

# Pour une mise en page large — doit être EN PREMIER avant tout st.
st.set_page_config(layout="wide")

# 0. Chargement des fichiers volumineux
# @st.cache_data est obligatoire ici : sans lui, Streamlit recharge le fichier
# à chaque interaction utilisateur (changement de film), ce qui provoque un crash mémoire.
# La fonction est nécessaire car @st.cache_data ne fonctionne que sur des fonctions.
@st.cache_data #pour éviter de recharger les données dès qu'on change de film
def load_affichage():
    # Petit CSV (~5MB) contenant uniquement les infos d'affichage (titre, poster, résumé...)
    lien_df_affichage = "df_affichage.csv"
    return pd.read_csv(lien_df_affichage, sep=",")

@st.cache_data
def load_ML():
    # Chargement de la matrice X en format numpy
    # + léger qu'un CSV pandas, crucial pour rester sous la limite de 1GB de Streamlit Cloud
    # La fonction est nécessaire car @st.cache_data ne fonctionne que sur des fonctions
    url = "https://huggingface.co/.../X_matrix.npy"
    response = requests.get(url)
    return np.load(io.BytesIO(response.content))

@st.cache_resource # pour éviter de recharger le modèle dès qu'on change de film
def load_model():
    # Modèle NearestNeighbors pré-entraîné
    lien_modele_reco_V2 = "https://huggingface.co/datasets/Elisa-Guerin/modele_reco_V2/resolve/main/modele_reco_V2.joblib"
    response = requests.get(lien_modele_reco_V2)
    return joblib.load(io.BytesIO(response.content))

# 1. Authentification 

lesDonneesDesComptes = {
    'usernames': {
        'utilisateur': {
            'name': 'utilisateur',
            'password': 'utilisateurMDP',
            'email': 'utilisateur@gmail.com',
            'failed_login_attemps': 0,  # Sera géré automatiquement
            'logged_in': False,          # Sera géré automatiquement
            'role': 'utilisateur'
        },
        'root': {
            'name': 'root',
            'password': 'rootMDP',
            'email': 'admin@gmail.com',
            'failed_login_attemps': 0,  # Sera géré automatiquement
            'logged_in': False,          # Sera géré automatiquement
            'role': 'administrateur'
        }
    }
}

authenticator = Authenticate(
    lesDonneesDesComptes,  # Les données des comptes
    "cookie name",         # Le nom du cookie, un str au choix
    "cookie key",          # La clé du cookie, un str au choix
    30,                    # Le nombre de jours avant que le cookie expire
)
authenticator.login()

# ── Gestion des statuts de connexion 
if st.session_state["authentication_status"] is False:
    st.error("L'username ou le password est/sont incorrect")

elif st.session_state["authentication_status"] is None:
    st.warning('Les champs username et mot de passe doivent être remplis')

elif st.session_state["authentication_status"]:  # utilisateur connecté

    # Le bouton de déconnexion
    authenticator.logout("Déconnexion")
# on indente tout notre sit a l'intérieur des conditions de log sinon le site s'affichera sans autentification)

    # Appels des fonctions de chargement — les données sont mises en cache au premier chargement
    # et réutilisées sans re-téléchargement à chaque interaction utilisateur
    df_affichage = load_affichage() # données légères pour l'affichage
    st.write(f"✅ CSV chargé : {df_affichage.shape}")
    df_ML = load_ML() # données encodées pour le KNN

    # Création de la matrice de features pour le modèle KNN
    # on part de la colonne 1 (index 1) pour exclure la colonne ID qui n'est pas une feature ML
    X = df_ML.iloc[:, 1:]  # on saute la colonne ID
    

    # ajout du modèle entrainé
    model4 = load_model()
    st.write(f"✅ Modèle chargé")

    # création du X_final avec le modèle entrainé + la colonne Titre de film
    X_final = X.copy()
    X_final["Titre"] = df_affichage["Title"]
    X_final["TitreAnnee"] = df_affichage["Title"] + " (" + df_affichage["Year"].astype(str) + ")"
    X_final["ID"] = df_ML["ID"]

    
    # Streamlit : création du site
    afficher_bandeau()  # bandeau hero Wildflix (remplace image_Netflix.webp)
    film_choisi = st.selectbox("Indique le film dont tu veux recevoir des recommandations", X_final["TitreAnnee"])
    id_film_choisi = X_final.loc[X_final["TitreAnnee"] == film_choisi, "ID"].values[0]

# 2. Tadame de Netflix
    def autoplay_audio(filepath: str):
        with open(filepath, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f"""
            <audio autoplay>
                <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            </audio>
        """, unsafe_allow_html=True)

    autoplay_audio("Netflix.mp3")
    
# 3. création des colonnes pour pouvoir centrer l'image
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        # Récupère l'affiche en fonction de l'ID (car titres en doublons et index différents selon les df)
        image = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Poster"].values[0] 
        st.image(image)
    with col2:
        resume_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Plot"].iloc[0]
        # Test traduction résumé sinon on affiche la caractéristique de base
        try :
            resume_choix_trad = GoogleTranslator(source="en", target="fr").translate(resume_choix)
            st.write(f"📖 Résumé : {resume_choix_trad}")
        except :
            st.write(f"📖 Résumé : {resume_choix}")
        
        # Test traduction genre sinon on affiche la caractéristique de base
        genre_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Genre"].iloc[0]
        try :
            genre_traduit = GoogleTranslator(source="en", target="fr").translate(genre_choix)
            st.write(f"🎞️ Genre : {genre_traduit}")
        except : 
            st.write(f"🎞️ Genre : {genre_choix}")
        
        annee_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Year"].iloc[0]
        st.write(f"📅 Année : {annee_choix}")
        duree_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Runtime"].iloc[0]
        st.write(f"⏳ Durée : {duree_choix}")
        acteurs_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Actors"].iloc[0]
        st.write(f"🎭 Acteurs : {acteurs_choix}")
        realisateur_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Director"].iloc[0]
        st.write(f"🎬 Réalisateur : {realisateur_choix}")
        note_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "imdbRating"].iloc[0]
        st.write(f"⭐ Note : {note_choix}/10")
        rated_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Rated"].iloc[0]
        st.write(f"🔞 Rated : {rated_choix}")
        nomination_choix = df_affichage.loc[df_affichage["ID"] == id_film_choisi, "Awards"].iloc[0]
        if pd.notna(nomination_choix):
            # Test traduction nomination sinon on affiche la caractéristique de base
            try : 
                nomination_traduit = GoogleTranslator(source="en", target="fr").translate(nomination_choix)
                st.write(f"🏆 Nomination : {nomination_traduit}")
            except : 
                st.write(f"🏆 Nomination : {nomination_choix}")


    #on ne veut garder que la ligne dont le film est écrit
    # .values transforme le df en array numpy, et le drop supprime la colonne titre pour le ML
    caracteristique_film_choisi = X_final[X_final["ID"] == id_film_choisi].drop(columns=["Titre", "ID", "TitreAnnee"]).values 

    # le [1] correpond à l'index des films recommandés, le [0] correspond à la distance depuis le film choisi
    # le [0] correcpond au premier crochet du deuxième array
    # le [1:] correspond à la liste des films recommandés en excluant le film de départ.
    # index_des_reco = (array([[0.        , 2.44948974, 2.44948974, 2.82842712, 2.82842712,
    # 2.82842712, 2.82842712, 2.82842712, 2.82842712, 2.82842712,
    # 2.82842712]]), array([[  0, 131, 207,  43,  50,  47,  71,  48,   1,  17,  12]]))  
    index_des_reco = model4.kneighbors(caracteristique_film_choisi)[1][0][1:]


    # on filtre les 10 resultats obtenus pour ne garde que les 5 mieux notés 
    top5reco = df_affichage.iloc[index_des_reco][["Title", "Year", "imdbRating", "ID"]].sort_values('imdbRating', ascending=False).head(5)
    top5 = top5reco.loc[:, "ID"].values
    top5_titre = top5reco.loc[:, "Title"]

    st.subheader(f"Comme tu as aimé {film_choisi}, je te recommande : ")

    # Créer les tabs avec les titres des films
    tabs = st.tabs(top5_titre.tolist())

    # CSS pour grossir le texte des onglets
    st.markdown("""
        <style>
            button[data-baseweb="tab"] p {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Itérer sur chaque tab et son film correspondant
    for tab, titre, id_film in zip(tabs, top5_titre, top5):
        # Récupère toutes les infos en une seule ligne
        film_info = df_affichage.loc[df_affichage["ID"] == id_film].iloc[0]

        image = film_info["Poster"]
        resume = film_info["Plot"]
        note = film_info["imdbRating"]
        realisateur = film_info["Director"]
        acteurs = film_info["Actors"]
        rated = film_info["Rated"]
        annee = film_info["Year"]
        nomination = film_info["Awards"]
        duree = film_info["Runtime"]
        genre = film_info["Genre"]

        with tab:
            col1, col2 = st.columns([1, 3], vertical_alignment="center")
            with col1:
                st.image(image)
            with col2:
                # Test traduction résumé sinon on affiche la caractéristique de base
                try : 
                    resume_traduit = GoogleTranslator(source="en", target="fr").translate(resume)
                    st.write(f"📖 Résumé : {resume_traduit}")
                except : 
                    st.write(f"📖 Résumé : {resume}")

                # Test traduction genre sinon on affiche la caractéristique de base
                try :    
                    genre_traduit = GoogleTranslator(source="en", target="fr").translate(genre)
                    st.write(f"🎞️ Genre : {genre_traduit}")
                except : 
                    st.write(f"🎞️ Genre : {genre}")
                
                st.write(f"📅 Année : {annee}")
                st.write(f"⏳ Durée : {duree}")
                st.write(f"🎭 Acteurs : {acteurs}")
                st.write(f"🎬 Réalisateur : {realisateur}")
                st.write(f"⭐ Note : {note}/10")
                st.write(f"🔞 Rated : {rated}")
                if pd.notna(nomination):
                    # Test traduction nomination sinon on affiche la caractéristique de base
                    try :
                        nomination_traduit = GoogleTranslator(source="en", target="fr").translate(nomination)
                        st.write(f"🏆 Nomination : {nomination_traduit}")
                    except : 
                        st.write(f"🏆 Nomination : {nomination}")
