# Application de Recommandation de Contenu

## Description

Cette application propose des recommandations d'articles personnalisées en fonction de l'identifiant utilisateur fourni. Elle utilise une combinaison de **filtrage collaboratif (CF)** et de **filtrage basé sur le contenu (CB)** pour générer des suggestions pertinentes.

## Fonctionnalités

- **Recommandations Personnalisées** : Obtenez une liste d'articles recommandés en fonction de votre historique et de vos préférences.
- **Interface Utilisateur** : Une interface simple et intuitive développée avec **Streamlit**.
- **Déploiement Cloud** : L'application est déployée en utilisant **Azure Functions** pour assurer scalabilité et performance.

## Prérequis

- **Python 3.9 ou supérieur**
- **Compte Azure** (optionnel, uniquement pour le déploiement de la fonction Azure)

## Installation

### 1️. Cloner le dépôt

```bash
git clone https://github.com/Krock13/AI_Engineer_Projet_10_Realisez-une-application-de-recommandation-de-contenu.git
cd AI_Engineer_Projet_10_Realisez-une-application-de-recommandation-de-contenu
```

### 2️. Créer un environnement virtuel et installer les dépendances

```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
```

### 3️. Télécharger les données

Les données doivent être téléchargées depuis Kaggle : 🔗 [Lien vers le dataset](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom/data)

- Dézipper l'archive `.zip`
- Placer les fichiers **décompressé** dans un dossier `data` à la racine du projet.

## Utilisation

### 1️. Lancer l'application Streamlit

```bash
streamlit run application/app.py
```

### 2️. Utilisation de l'application

- Entrez un **identifiant utilisateur** dans le champ prévu à cet effet.
- Cliquez sur le bouton **"Obtenir des recommandations"**.
- L'application affichera une sélection d'articles recommandés, générés par **la fonction Azure déjà déployée sur le cloud**.

## Déploiement Azure (Facultatif)

**Le déploiement de la fonction Azure n'est pas nécessaire, car elle est déjà active sur le cloud.** Cependant, si vous souhaitez redéployer la fonction Azure :

### 1️. Naviguer vers le répertoire de la fonction Azure

```bash
cd Azure_function
```

### 2️. Publier la fonction

Suivez les instructions officielles de **Microsoft** pour déployer une **Azure Function en Python**. Assurez-vous d'avoir configuré votre environnement Azure et d'avoir les permissions nécessaires.
