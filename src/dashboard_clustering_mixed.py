import streamlit as st
import pandas as pd
import plotly.express as px
from constant import PATH_DATA, PATH_OUTPUT
from streamlit_plotly_events import plotly_events
import plotly.express as px
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
 

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

cmap = plt.cm.get_cmap("tab20c", 100)  # Using the "tab20c" colormap
colors = [rgb_to_hex(cmap(i)) for i in range(100)]




def load_images_as_array(image_paths, target_size=(128, 128)):
    """
    Load images from file paths and return a NumPy array.
    
    Parameters:
    - image_paths: Pandas Series or list containing file paths to images.
    - target_size: Tuple (width, height) to resize images. Default is (128, 128).
    
    Returns:
    - A NumPy array of shape (num_images, height, width, channels)
    """
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)  # Read image
        if img is None:
            print(f"Warning: Could not read {path}")
            continue
        img = cv2.resize(img, target_size)  # Resize to target size
        images.append(img)

    return images

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


kValues = [5, 10, 15,20, 25]
silouhettes = np.load(os.path.join(PATH_OUTPUT, "silhouette_scores.npy"))
inerties = np.load(os.path.join(PATH_OUTPUT, "inertie_scores.npy"))
def plot_silouhette(silouhettes, kvalues):
    """
    Plot the silhouette scores for different values of k using Plotly in Streamlit.
    
    Parameters:
    - silouhettes: list or array of silhouette scores
    - kvalues: list of corresponding k values
    """
    if len(silouhettes) != len(kvalues):
        st.error("Erreur : silouhettes et kvalues doivent avoir la même longueur.")
        return

    df_silhouette = pd.DataFrame({
        'k': kvalues,
        'silhouette_score': silouhettes
    })

    fig = px.line(
        df_silhouette, 
        x='k', 
        y='silhouette_score', 
        markers=True,
        title="Évolution des scores de silhouette en fonction de k",
        labels={'k': 'Nombre de clusters (k)', 'silhouette_score': 'Score de silhouette'},
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_inerties(inerties, kvalues):
    """
    Plot the inertia scores for different values of k using Plotly in Streamlit.
    
    Parameters:
    - inerties: list or array of inertia scores
    - kvalues: list of corresponding k values"""
    if len(inerties) != len(kvalues):
        st.error("Erreur : inerties et kvalues doivent avoir la même longueur.")
        return

    df_inertie = pd.DataFrame({
        'k': kvalues,
        'inertie_score': inerties
    })

    fig = px.line(
        df_inertie, 
        x='k', 
        y='inertie_score', 
        markers=True,
        title="Évolution des scores d'inertie en fonction de k",
        labels={'k': 'Nombre de clusters (k)', 'inertie_score': 'Score d\'inertie'},
    )

    st.plotly_chart(fig, use_container_width=True)

def display_cluster_images(cluster_indices, images, descriptor, selected_cluster, images_per_row=5):
    st.write(f"### Exemples d'images du Cluster {selected_cluster} ({descriptor}) : {len(cluster_indices)}")

    if len(cluster_indices) == 0:
        st.write("⚠️ Aucun exemple d'image trouvé pour ce cluster.")
        return

    # Calculate the number of rows needed
    num_images = len(cluster_indices)
    num_rows = (num_images // images_per_row) + (1 if num_images % images_per_row != 0 else 0)

    # Display images in a grid
    for row in range(num_rows):
        cols = st.columns(images_per_row)  # Create a row of columns
        for col in range(images_per_row):
            img_idx = row * images_per_row + col
            if img_idx < num_images:  # Check if there's an image to display
                img = images[cluster_indices[img_idx]][..., ::-1]  # Récupérer l'image
                cols[col].image(img, use_column_width=True)  # Display the image in the column


def colorize_cluster(cluster_data, selected_cluster):
    # Ensure 'cluster' column exists
    if 'cluster' not in cluster_data.columns: raise ValueError("'cluster' column not found in data")
    
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster', color_continuous_scale='Viridis')

    # If a specific cluster is selected, filter for that cluster
    if selected_cluster != "All Clusters":
        filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
        selected_color = colors[int(selected_cluster) % len(colors)] 

        fig.add_scatter3d(
            x=filtered_data['x'], 
            y=filtered_data['y'], 
            z=filtered_data['z'],
            mode='markers',
            marker=dict(color=selected_color, size=10),  # Red points for selected cluster
            name=f'Selected Cluster {selected_cluster}',
            customdata=filtered_data['image']
        )

    else:
        filtered_data = cluster_data
        selected_color = 'blue'

    return fig



descriptors = []

df_mixed=[]
df_metric_mixed=[]
df_clip=[]
df_metric_clip=[]
df_deepcluster=[]
df_metric_deepcluster=[]
df_sim=[]
df_metric_sim=[]



try:
    df_mixed = pd.read_excel(PATH_OUTPUT +"/save_clustering_mixed.xlsx")
    df_metric_mixed = pd.read_excel(PATH_OUTPUT +"/save_metric_mixed.xlsx")
    descriptors.append("Merged Features")
    if 'Unnamed: 0' in df_metric_mixed.columns: df_metric_mixed.drop(columns="Unnamed: 0", inplace=True)
except Exception as e:
    print(f"Error loading mixed features files: {str(e)}")


try:
    df_clip = pd.read_excel(PATH_OUTPUT+"/save_clustering_clip.xlsx")
    df_metric_clip = pd.read_excel(PATH_OUTPUT+"/save_metric_clip.xlsx")
    descriptors.append("clip")
    if 'Unnamed: 0' in df_metric_clip.columns: df_metric_clip.drop(columns="Unnamed: 0", inplace=True)
except Exception as e:
    print(f"Error loading DeepCluster files: {str(e)}")


# Process DeepCluster
try:
    df_deepcluster = pd.read_excel(PATH_OUTPUT+"/save_clustering_deepcluster.xlsx")
    df_metric_deepcluster = pd.read_excel(PATH_OUTPUT+"/save_metric_deepcluster.xlsx")
    descriptors.append("resnet")
    if 'Unnamed: 0' in df_metric_deepcluster.columns: df_metric_deepcluster.drop(columns="Unnamed: 0", inplace=True)
except Exception as e:
    print(f"Error loading DeepCluster files: {str(e)}")


try:
    df_sim = pd.read_excel(PATH_OUTPUT+"/save_clustering_SIM.xlsx")
    df_metric_sim = pd.read_excel(PATH_OUTPUT+"/save_metric_SIM.xlsx")
    descriptors.append("self_supervised")
    if 'Unnamed: 0' in df_metric_sim.columns: df_metric_sim.drop(columns="Unnamed: 0", inplace=True)
except Exception as e:
    print(f"Error loading sim files: {str(e)}")



# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:
    st.write('## Résultat de Clustering des données DIGITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs

    #TODO ADD OTHER FEATURES INTO ACCOUNT
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', descriptors)
    if descriptor=="Merged Features": 
        df = df_mixed
        df_metric = df_metric_mixed
    elif descriptor=="self_supervised": 
        df = df_sim
        df_metric = df_metric_sim
    elif descriptor=="clip": 
        df = df_clip
        df_metric = df_metric_clip
    elif descriptor=="resnet": 
        df = df_deepcluster
        df_metric = df_metric_deepcluster  


    images = load_images_as_array(df["image"])

    unique_clusters = df['cluster'].unique().tolist()
    unique_clusters.sort()  # Sort the clusters for better readability
    unique_clusters.append("All Clusters")


    # Ajouter un sélecteur pour les clusters
    selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', unique_clusters, index=len(unique_clusters) - 1)
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index #afficher seulement images[cluster_indices]
    # Sélection du cluster choisi
    if selected_cluster == "All Clusters":
        filtered_data = df
    else:
        filtered_data = df[df['cluster'] == selected_cluster]
    
    fig = colorize_cluster(filtered_data, selected_cluster)
    

    fig.update_traces(
        hovertemplate="<b>X:</b> %{x}<br>"
                  "<b>Y:</b> %{y}<br>"
                  "<b>Z:</b> %{z}<br>"
                  "<b>Cluster:</b> %{marker.color}<br>"
                  "<b>Image:</b> %{customdata[0]}<extra></extra>"
                  "<extra></extra>"
    )

    
    #st.plotly_chart(fig, on_select="rerun", key="scatter_plot", use_container_width=True)
    selected_points = plotly_events(fig, select_event=True)
    if selected_points:
        point_number = selected_points[0]['pointNumber']
        st.write(f"Point Number: {point_number}")
        clicked_point_data = filtered_data.iloc[point_number]
        st.write("Image of the clicked point:")
        image_index = df[df['image'] == clicked_point_data['image']].index[0]
        # Retrieve the corresponding image from images
        image = images[image_index]
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
            image = image[..., ::-1]  # Convert from BGR to RGB

        # Convert NumPy array to PIL Image
        image_pil = Image.fromarray(image)
        
        cols = st.columns(5)
        with cols[0]:  
            st.image(image, caption="Clicked Image", width=150)
    
    display_cluster_images(cluster_indices, images, descriptor, selected_cluster)


# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Métriques ' )
    st.dataframe(df_metric)
    plot_silouhette(silouhettes, kValues) #commenter si on a a pas utilisé kmeans
    plot_inerties(inerties, kValues) #commenter si on a pas utilisé kmeans

