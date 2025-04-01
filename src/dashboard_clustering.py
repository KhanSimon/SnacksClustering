
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from features import *
from constant import PATH_DATA



@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    
    fig = px.bar(
        df_metric, 
        x='descriptor', 
        y='ami',
        color='descriptor',
        color_discrete_sequence=['#1f77b4', '#ff7f0e'],
        title='Comparaison des scores AMI',
    )
    st.plotly_chart(fig)

def display_cluster_images(cluster_indices, images, descriptor, selected_cluster):
    st.write(f"### Exemples d'images du Cluster {selected_cluster} ({descriptor})")

    if len(cluster_indices) == 0:
        st.write("⚠️ Aucun exemple d'image trouvé pour ce cluster.")
        return

    num_images = min(len(cluster_indices), 10)  # Afficher max 10 images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Assurer que axes est une liste même si une seule image
    if num_images == 1:
        axes = [axes]

    for idx, img_idx in enumerate(cluster_indices[:num_images]):
        img = images[img_idx][..., ::-1]   # Récupérer l'image depuis le tableau
        axes[idx].imshow(img)
        axes[idx].axis("off")

    st.pyplot(fig)




 
# Charger les images UNE SEULE FOIS en dehors de la fonction
images, _ = load_images_and_labels(PATH_DATA)
# Chargement des données du clustering
df_hog = pd.read_excel("../output/save_clustering_hog_kmeans.xlsx")
df_deep = pd.read_excel("../output/save_clustering_deep_kmeans.xlsx")
df_combined = pd.read_excel("../output/save_clustering_combined_kmeans.xlsx")
df_metric = pd.read_excel("../output/save_metric.xlsx")

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données DIGITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["HOG","DEEP","COMBINED"])
    if descriptor=="HOG":
        df = df_hog
    if descriptor=="DEEP":
        df = df_deep
    if descriptor=="COMBINED":
        df = df_combined
    # Ajouter un sélecteur pour les clusters
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(21), index = 20)
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    #afficher seulement images[cluster_indices]
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    if (selected_cluster == 20): 
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster')
    else : 
        fig = px.scatter_3d(filtered_data, x='x', y='y', z='z', color='cluster')
    st.plotly_chart(fig)
    display_cluster_images(cluster_indices, images, descriptor, selected_cluster)

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Métriques ' )
    # TODO :à remplir par un affichage d'un tableau
    st.dataframe(df_metric)