from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, accuracy_score
from sklearn import metrics

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,n_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            n_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    tsne = TSNE(n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                n_iter=n_iter
               )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label,l_cluster):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    return df




def clustering_accuracy(labels_true, cluster_labels):
    """
    Calcule l'accuracy de clustering (non supervisée).
    On fait correspondre chaque cluster à la classe majoritaire.
    Ensuite, on compte la proportion de prédictions justes.
    Utilise l'algorithme de Kuhn-Munkres (Hungarian method).
    """
    # On construit la matrice de confusion
    # rows = vraies classes, cols = clusters
    cm = confusion_matrix(labels_true, cluster_labels)
    
    # On cherche l'assignation (classe -> cluster) qui maximise les bonnes prédictions
    # Hungarian method renvoie une permutation optimal.
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Sommation des éléments sur la correspondance trouvée
    total_correct = cm[row_ind, col_ind].sum()
    accuracy = total_correct / np.sum(cm)
    return accuracy



def show_metric(labels_true, labels_pred, descriptors,bool_return=False,name_descriptor="", name_model="kmeans",bool_show=True):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    accuracy = clustering_accuracy(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"ARI : Adjusted Rand Index: {ari}")
        print(f"AMI : Adjusted Mutual Information: {ami}")
        print(f"ACCURACY :Accuracy score: {accuracy}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model,
               "accuracy":accuracy}