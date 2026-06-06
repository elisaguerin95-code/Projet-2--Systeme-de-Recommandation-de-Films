# 🎬 Système de Recommandation de Films

Application de recommandation de films basée sur le Machine Learning (algorithme KNN), enrichie avec les données de l'API OMDB.

---

## 📋 Description

À partir d'un dataset de films, ce projet récupère des données enrichies via l'API OMDB (genre, acteurs, réalisateur, note IMDB, affiche...), puis entraîne un modèle KNN pour recommander 5 films similaires à celui choisi par l'utilisateur.

---

## 🌐 Application en ligne

👉 [Accéder à l'application](https://projet-2--systeme-de-recommandation-de-films.streamlit.app/)

---

## 📁 Structure du projet

| Fichier | Description |
|---------|-------------|
| `preparation_data_projet2.py` | Préparation des données et entraînement du modèle ML |
| `projet2_V2.py` | Application Streamlit de recommandation |
| `bandeau.py` | Module d'affichage du bandeau Wildflix |
| `Netflix.mp3` | Son de démarrage |
| `requirements.txt` | Librairies nécessaires |
| `data_patrickV2.csv` | Dataset source |
| `df_affichage.csv` | Dataset léger pour l'affichage (généré) |
| `X_matrix.npy` *(Hugging Face)* | Matrice de features KNN encodées en int8 (~40MB) |
| `modele_reco_V2.joblib` *(Hugging Face)* | Modèle KNN entraîné (généré) |

---

## ⚙️ Comment ça marche ?

### Fichier 1 : `preparation_data_projet2.py`

Ce fichier prépare les données en 8 étapes :

1. **Import du dataset** et création d'une colonne ID à partir des liens IMDB
2. **Récupération des données API** — le dataset est divisé en 6 tranches de 1000 films (limite de 1000 requêtes/jour par clé API OMDB)
3. **Concaténation** des 6 extractions API en un seul DataFrame
4. **Fusion** du dataset original avec les données API via l'ID IMDB
5. **Filtrage** — conservation uniquement des films (suppression des séries et jeux)
6. **Sélection des colonnes** utiles (suppression des colonnes non pertinentes)
7. **Nettoyage** — standardisation de la colonne `Rated`, complétion des valeurs manquantes
8. **Création du DataFrame ML et entrainement du modèle KNN** — encodage des genres, acteurs et réalisateurs en colonnes binaires, export du modèle

⚠️ **Note** : Si vous avez déjà exécuté la partie API, vous pouvez repartir directement du fichier `data_API_V2.csv` à l'étape 4.

### Fichier 2 : `projet2_V2.py`

Application Streamlit qui :
- Gère l'authentification des utilisateurs avec (`streamlit-authenticator`)
- Affiche un bandeau **Wildflix** et joue la musique de démarrage Netflix
- Charge le dataset d'affichage (`df_affichage.csv`) depuis GitHub, la matrice de features (`X_matrix.npy`) et le modèle entraîné (`modele_reco_V2.joblib`) depuis Hugging Face
- Propose un menu de sélection de film (titre + année pour différencier les homonymes)
- Affiche le poster, le résumé traduit en français, le genre, la durée, les acteurs, le réalisateur, la note et les nominations du film choisi
- Recommande 5 films similaires (filtrés parmi les 10 plus proches par note IMDB), affichés dans des onglets avec toutes leurs informations

---

## 🚀 Installation et utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

### Étape 1 — Préparer les données

1. Obtenir une clé API gratuite sur [OMDB API](http://www.omdbapi.com/apikey.aspx)
2. Renseigner votre clé dans `preparation_data_projet2.py` :
```python
cle_API = "VOTRE_CLE_API"
```
3. Exécuter le fichier (**attention** : lancer les parties API une par une, 1000 films/jour maximum)

### Étape 2 — Lancer l'application

```bash
streamlit run projet2_V2.py
```

**Identifiants de connexion :**
- Utilisateur : `utilisateur` / Mot de passe : `utilisateurMDP`
- Administrateur : `root` / Mot de passe : `rootMDP`

---

## 👥 Auteurs

Projet réalisé par Cédric C., Claire G., Elisa G., Eloïse G.
