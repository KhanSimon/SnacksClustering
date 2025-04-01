import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools
from sklearn.cluster import KMeans as KMeansSL
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_images_and_labels(data_dir):
    images = []  # List to store all images
    labels = []  # List to store labels for each image
    
    # Get the names of the subdirectories (apple, banana, candy)
    categories = ['apple', 'banana', 'cake', 'candy', 'carrot', 'cookie', 'doughnut', 'grape', 'hot dog', 'ice cream','juice','muffin','orange','pineapple','popcorn','pretzel','salad','strawberry','waffle','watermelon']
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        
        # Check if the subdirectory exists
        if not os.path.isdir(category_dir):
            print(f"Directory {category_dir} not found.")
            continue
        
        # Iterate through the images in each subdirectory
        for filename in os.listdir(category_dir):
            image_path = os.path.join(category_dir, filename)
            
            # Check if it's a valid image file (optional check for specific file types)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)
                
                # Optionally resize the image to a consistent size (e.g., 128x128)
                image = cv2.resize(image, (128, 128))
                
                # Append the image and its corresponding label
                images.append(image)
                labels.append(label)  # label is 0 for apple, 1 for banana, 2 for candy
    
    # Convert images and labels to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    return images_array, labels_array








def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:#chaque pixel est codé sur 4 bits avec 16 niveaux de gris possibles
        #la fonction cv2.calcHist attends des images avec des pixels sur 8 bits : 256 niveaux possibles
        #il faut convertir : 
        image = image.astype(np.uint8)  
        
        descriptor = cv2.calcHist([image],  #on envoie l'image sous forme de liste
                                  channels=[0],  # canal unique car niveaux de gris
                                  mask=None,  #pas de masque
                                  histSize=[16],  #16 bins pour 16 niveaux de gris
                                  ranges=[0, 16])  #plage des niveaux de gris (0-15)

        descriptors.append(descriptor.flatten())  # on converti en 1D
    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    for image in images:
        descriptor = hog(image, 
                         orientations=8,  # on effectue le gradient dans toutes les directions, diagonales comprises
                         pixels_per_cell=(2, 2),  #on créé des cellules (ensembles de pixels) de 4 pixels
                         cells_per_block=(1, 1),  # pas de normalisation par blocs, chaque cellule est indépendante
                         visualize=False, # pas besoin de visualisation
                         channel_axis = -1)  
        descriptors.append(descriptor)
    return descriptors

def compute_color_histograms(images, color_space="RGB"):
    """
    Calcule les histogrammes de couleurs pour des images en 3D.
    Input : 
        - images (list) : liste des images en couleur (BGR sous OpenCV)
        - color_space (str) : "RGB" ou "HSV" pour définir l'espace de couleur utilisé
    Output : 
        - descriptors (list) : liste des descripteurs d'histogrammes concaténés
    """
    descriptors = []
    
    for image in images:
        # Conversion de l'espace colorimétrique si nécessaire
        if color_space == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("color_space doit être 'RGB' ou 'HSV'")

        hist_features = []
        
        for channel in range(3):  # Boucle sur chaque canal (R, G, B) ou (H, S, V)
            hist = cv2.calcHist([image],  # Liste contenant l'image
                                channels=[channel],  # Canal spécifique (0, 1 ou 2)
                                mask=None,  # Pas de masque
                                histSize=[16],  # 16 bins comme pour l'image en niveaux de gris
                                ranges=[0, 256])  # Plage des valeurs des pixels
            
            hist_features.append(hist.flatten())  # Converti en vecteur 1D
        
        # Concaténation des histogrammes des 3 canaux
        descriptors.append(np.concatenate(hist_features))
    
    return descriptors


def compute_sift_descriptor(images):
    """
    Applique SIFT sur une liste d'images et renvoie une matrice (nb_images, 128).
    Chaque image est représentée par la moyenne de ses descripteurs SIFT.

    Input :
        - images (list) : liste d'images en niveaux de gris (128x128)
    
    Output :
        - sift_features (ndarray) : tableau de taille (nb_images, 128)
    """
    sift = cv2.SIFT_create()
    descriptors_list = []

    for image in images:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            mean_descriptor = np.mean(descriptors, axis=0)  # Moyenne des descripteurs
        else:
            mean_descriptor = np.zeros(128)  # Si aucun point détecté, vecteur nul

        descriptors_list.append(mean_descriptor)

    return np.array(descriptors_list)

def compute_bovw_features(images, n_clusters=100):
    """
    Crée un Bag of Visual Words (BoVW) basé sur SIFT et KMeans.
    Renvoie un tableau de taille (nb_images, n_clusters).

    Input :
        - images (list) : liste d'images en niveaux de gris
        - n_clusters (int) : nombre de clusters pour KMeans

    Output :
        - bovw_features (ndarray) : histogramme des visuels words (nb_images, n_clusters)
    """
    sift = cv2.SIFT_create()
    all_descriptors = []

    for image in images:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            all_descriptors.append(descriptors)

    # Concaténer tous les descripteurs en une seule matrice
    all_descriptors = np.vstack(all_descriptors)

    # Appliquer KMeans pour créer un dictionnaire de visual words
    kmeans = KMeansSL(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    # Créer l'histogramme BoVW pour chaque image
    bovw_features = np.zeros((len(images), n_clusters))

    for i, image in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            for label in labels:
                bovw_features[i, label] += 1  # Compte les occurrences de chaque cluster

    return bovw_features

def compute_deep_features(images):
    """
    Extrait des features d'un tableau d'images en utilisant ResNet50 pré-entraîné sur ImageNet.
    
    Paramètres:
        images (numpy array): Tableau d'images de forme (N, 128, 128, 3)
        
    Retourne:
        numpy array: Features extraites de forme (N, n_features)
    """
    # Vérification de la forme du tableau
    assert len(images.shape) == 4, "Le tableau doit être de forme (N, 128, 128, 3)"

    # Charger ResNet50 sans la dernière couche (Fully Connected)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Ajouter un GlobalAveragePooling pour réduire la dimension des features
    model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

    # Normalisation des images pour ResNet50
    images = preprocess_input(images)

    # Extraire les features
    features = model.predict(images)

    return features  # Retourne un tableau de (955, n_features)





