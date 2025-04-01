from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn.decomposition import PCA
import umap
import numpy as np
import pickle
from sklearn.preprocessing import normalize
#--------------------------------------------------
from preparation import *
from utils import *
from constant import PATH_OUTPUT, PATH_DATA, CACHE_FILE
#--------------------------------------------------

from sklearn.cluster import KMeans as KMeansSL
from hdbscan import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
#--------------------------------------------------



def save_features(feature_list):
    """Saves the result of feature extracting from images. These calculations take time and this allows us to avoid redoing them every time"""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(feature_list, f)

def load_features():
    """loads the result of feature extracting from images."""
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)
    

def reassign_noise_points(features, labels):
    """When doing HDBscan. some images remain unclssified because they do not belong to a region whit enough density of points. 
    so they are put in a (-1) cluster which indicates unspecified. This method classify these points into the nearest 
    cluster"""
    non_noise = labels != -1
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features[non_noise], labels[non_noise])
    for i, label in enumerate(labels):
        if label == -1:
            labels[i] = knn.predict([features[i]])[0]  # Assign nearest cluster
    return labels


def compute_cluster_centroids(features, labels):
    """Compute the centroid of each cluster."""
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        if label == -1: continue
        cluster_points = features[labels == label]
        centroid = np.mean(cluster_points, axis=0)  # Compute mean feature vector
        centroids[label] = centroid

    return centroids


def merge_clusters_with_kmeans(cluster_centroids, target_clusters=20):
    """Force merge clusters into exactly 'target_clusters' using K-Means.
    It uses kmeans to construct target_clusters applied on the centroied of
    the previous clusters. it returns the lable for each centroid"""
    centroids_array = np.array(list(cluster_centroids.values()))
    kmeans = KMeansSL(n_clusters=target_clusters, init='k-means++', n_init=10)
    merged_labels = kmeans.fit_predict(centroids_array)
    return merged_labels



def relabel_images(dbscan_labels, merged_labels):
    """Map original labels to merged  lables."""
    mapping = {old: new for old, new in zip(np.unique(dbscan_labels), merged_labels)}
    new_labels = np.array([mapping[label] if label in mapping else -1 for label in dbscan_labels])
    return new_labels




    


def pipeline(use_features=None, feature_weights=None, num_clusters=20, visiualisation=False, reduceDemansion = True,clusteringAlgo = "kmeans",reduceDemansionAlgo = "pca",targetDimansions = 50, force_clusters=False):
    if use_features is None:
        use_features = {"hog": True,"histogram": True,"color_stats": True,"sift": True,"contour": True,"cnnFood":True,"deepcluster":True,"clip":True}

    if feature_weights is None:
        feature_weights = {"hog": 1.0,"histogram": 1.0,"color_stats": 1.0,"sift": 1.0,"contour": 1.0,"cnnFood":1.0,"deepcluster":1.0,"clip":1.0}

    images, labels_true,paths = load_images_and_labels(PATH_DATA,withPaths=True)
    
    print("\n\n ##### Feature Extraction ######")
    feature_list = []
    
    if os.path.exists(CACHE_FILE):
        print("- Loading features from cache...")
        feature_list = load_features()
    else:
        if use_features.get("clip", False):
            print("- Extracting clip features...")
            descriptors_clip = extract_clip_features_batch(images)
            feature_list.append((descriptors_clip, "clip")) 
           
        if use_features.get("hog", False):
            print("- Extracting HOG features...")
            descriptors_hog = extract_hog_features_batch(images)
            feature_list.append((descriptors_hog, "hog"))

        if use_features.get("histogram", False):
            print("- Extracting Color Histogram features...")
            descriptors_hist = extract_color_histogram_batch([normalize_color(image) for image in images])
            feature_list.append((descriptors_hist, "histogram"))

        if use_features.get("color_stats", False):
            print("- Extracting Color Statistics features...")
            descriptors_color_stats = extract_color_stats_batch([normalize_color(image) for image in images])
            feature_list.append((descriptors_color_stats, "color_stats"))

        if use_features.get("sift_mean", False) or use_features.get("sift_bow", False) :
            print("- Extracting SIFT features...")
            descriptors_mean_sift,descriptors_bow_sift = compute_sift_representations(images)
            feature_list.append((descriptors_mean_sift, "sift_mean"))
            feature_list.append((descriptors_bow_sift, "sift_bow"))

        if use_features.get("contour", False):
            print("- Extracting Contour features...")
            descriptors_contour = extract_contour_features_batch(images)
            feature_list.append((descriptors_contour, "contour"))  

        if use_features.get("deepcluster", False):
            print("- Extracting deepcluster features...")
            descriptors_deepcluster = extract_deepcluster_features_batch(images)
            feature_list.append((descriptors_deepcluster, "deepcluster"))  

            

        print("- Computing and saving features...")
        save_features(feature_list)   


    print("\n\n ##### Data Preprocessing & Feature Weighting ######")
    scaler = StandardScaler()
    all_features = []


    for descriptor, name in feature_list:
        if use_features.get(name, False) and feature_weights[name]>0:
            scaled_descriptor = scaler.fit_transform(descriptor) * feature_weights[name]
            all_features.append(scaled_descriptor)

    # Merge all selected features
    all_features = np.hstack(all_features)
    

    print("\n\n ##### Data dimension reduction ######")
    if reduceDemansion :
        if reduceDemansionAlgo == "pca" :
            pca = PCA(n_components=targetDimansions)  # Reduce to 50 dimensions
            all_features_reduced = pca.fit_transform(all_features)
        elif reduceDemansionAlgo == "umap" :
            umap_model = umap.UMAP(n_components=targetDimansions, metric="cosine")
            all_features_reduced = umap_model.fit_transform(all_features)   

    else : all_features_reduced = all_features 

    all_features_reduced = normalize(all_features_reduced, axis=1)

    print("shuffling...")
    shuffle_images_and_labels(images, labels_true, all_features_reduced,paths)       
    
    
    print(f"\n\n ##### Clustering with {clusteringAlgo} ######")
    if clusteringAlgo=="kmeans" :
        kValues = [5, 10, 15, 25]
        silhouette_scores = []
        inertie_scores = []
        for k in kValues:
            clustering_model = KMeansSL(n_clusters=k, init='k-means++', n_init=10)
            clustering_model.fit(all_features_reduced)
            labels_pred = clustering_model.labels_
            metric = show_metric(labels_true,labels_pred , all_features_reduced, bool_show=False, name_descriptor="Merged Features", bool_return=True,name_model=clusteringAlgo)
            silhouette_scores.append(metric['silhouette'])
            inertie_scores.append(clustering_model.inertia_)
        
        #print("silouhettes avant",silhouette_scores)
        #print("ineties avant",inertie_scores)
        clustering_model = KMeansSL(n_clusters=num_clusters, init='k-means++', n_init=10)
        clustering_model.fit(np.array(all_features_reduced))
        labels_pred = clustering_model.labels_
        metric = show_metric(labels_true,labels_pred , all_features_reduced, bool_show=False, name_descriptor="Merged Features", bool_return=True,name_model=clusteringAlgo)
        silhouette_scores.insert(-1, metric['silhouette'])
        inertie_scores.insert(-1, clustering_model.inertia_)
        #print("ineties après",inertie_scores)
        #print("silouhettes après",silhouette_scores)

    elif clusteringAlgo=="hdbscan" :
        clusterer = HDBSCAN(min_cluster_size=14, min_samples=2, metric="euclidean")
        labels_pred = clusterer.fit_predict(all_features_reduced) 
        if force_clusters:
            print("\n\n ##### reassign unclassified points ######")
            labels_pred = reassign_noise_points(all_features_reduced, labels_pred)
            print("\n\n ##### ensure exactly n clusterss ######")
            #centroids = compute_cluster_centroids(all_features_reduced, labels_pred)
            #merged_cluster_labels = merge_clusters_with_kmeans(centroids, target_clusters=num_clusters)
            #labels_pred = relabel_images(labels_pred, merged_cluster_labels)
    elif clusteringAlgo=="agglomerative" :
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        labels_pred = clustering_model.fit_predict(all_features)        
    else :
        raise ValueError('clusteringAlfo not recognised')    


    

    print("\n\n ##### Results ######")
    print(f"\n\n ##### Clustering with Weights: hog={feature_weights['hog']},contour={feature_weights['contour']}, histogram={feature_weights['histogram']},color_stats={feature_weights['color_stats']}, sift_mean={feature_weights['sift_mean']}, sift_bow={feature_weights['sift_bow']},deepcluster={feature_weights['deepcluster']},clip={feature_weights['clip']} ######")
    metric = show_metric(labels_true,labels_pred , all_features_reduced, bool_show=True, name_descriptor="Merged Features", bool_return=True,name_model=clusteringAlgo)
    if visiualisation :
        print("- Exporting data for visualization")
        df_metric = pd.DataFrame([metric])
        # Convert to 3D for visualization
        x_3d = conversion_3d(all_features_reduced)
        df_export = create_df_to_export(x_3d, labels_true, labels_pred)
        df_export["image"] = paths
        # Ensure output directory exists
        os.makedirs(PATH_OUTPUT, exist_ok=True)
        # Save results
        active_features = [feature for feature, weight in feature_weights.items() if weight != 0]
        if len(active_features) == 1 and (active_features[0]=="clip" or active_features[0]=="deepcluster"):
            feature_name = active_features[0]
            clustering_filename = f"/save_clustering_{feature_name}.xlsx"
            metric_filename = f"/save_metric_{feature_name}.xlsx"
        else:
            clustering_filename = "/save_clustering_mixed.xlsx"
            metric_filename = "/save_metric_mixed.xlsx"


        df_export.to_excel(PATH_OUTPUT + clustering_filename)
        df_metric.to_excel(PATH_OUTPUT + metric_filename)
        print("Done! To visualize results, run: streamlit run dashboard_clustering_mixed.py")
        # Save the silhouette and inertia scores
        np.save(os.path.join(PATH_OUTPUT, "silhouette_scores.npy"), silhouette_scores)
        np.save(os.path.join(PATH_OUTPUT, "inertie_scores.npy"), inertie_scores)


if __name__ == "__main__":
    pipeline(
        use_features={"hog": False, "histogram": False, "color_stats": False, "contour": False, "sift_mean": False, "sift_bow": False,"deepcluster":True,"clip":True},
        feature_weights={"hog": 0, "histogram": 0, "color_stats": 0, "contour": 0, "sift_mean": 0, "sift_bow": 0,"deepcluster":1,"clip":0},
        visiualisation = True,#si on donne en sortie les fichiers xlsx pour la visualisation avec dashboard
        reduceDemansion = True,#si on reduit les dimensions
        targetDimansions = 10,#en cas de redemesnionnement,  
        reduceDemansionAlgo = "umap",#algorithme de redemensionnement
        clusteringAlgo = "kmeans",# cluster algorithm (kmeans, hdbscan, etc.)
        force_clusters=True #en cas de hdbscan, on attribue ou non les points dans le cluster -1 au cluster le plus proche
    )


