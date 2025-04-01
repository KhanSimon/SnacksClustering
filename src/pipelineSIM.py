
from sklearn.discriminant_analysis import StandardScaler
from utils import show_metric
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from constant import PATH_OUTPUT, PATH_DATA, PATH_DATA2, MODEL_CLUSTERING
from features import *
from sklearn.preprocessing import normalize
from hdbscan import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
import umap.umap_ as umap
from utils import *

images_array, labels_true,paths = load_images_and_labels(PATH_DATA2,withPaths=True)

all_embeddings = np.load(os.path.join(PATH_OUTPUT, "SimCLRfinal0.1.npy"))
print("Embeddings chargés :", all_embeddings.shape)

# ====================================================================== #
# 6) Réduction de dimension via PCA (50) + Clustering
# ====================================================================== #


# Réduction de dimension avec UMAP à 50 dimensions (meilleur que PCA)
umap_reducer = umap.UMAP(n_components=50, random_state=42)
embeddings_50UMAP = umap_reducer.fit_transform(all_embeddings)  # shape (N, 50)

embeddings_norm = normalize(embeddings_50UMAP, axis=1) #essayer sans et avec : meilleur score ami quand on normalise : + 1 

n_clusters = 20

kValues = [5, 10, 15, 25]
silhouette_scores = []
inertie_scores = []
for k in kValues:
    clustering_model = KMeansSL(n_clusters=k, init='k-means++', n_init=10)
    clustering_model.fit(embeddings_norm)
    labels_pred = clustering_model.labels_
    metric = show_metric(labels_true,labels_pred , embeddings_norm, bool_show=False, name_descriptor="Merged Features", bool_return=True,name_model="kmeans")
    silhouette_scores.append(metric['silhouette'])
    inertie_scores.append(clustering_model.inertia_)

print("silouhettes avant",silhouette_scores)
print("ineties avant",inertie_scores)
clustering_model = KMeansSL(n_clusters=n_clusters, init='k-means++', n_init=10)
clustering_model.fit(np.array(embeddings_norm))
labels_pred = clustering_model.labels_
metric = show_metric(labels_true,labels_pred , embeddings_norm, bool_show=False, name_descriptor="Merged Features", bool_return=True,name_model="kmeans")
silhouette_scores.insert(-1, metric['silhouette'])
inertie_scores.insert(-1, clustering_model.inertia_)
print("ineties après",inertie_scores)
print("silouhettes après",silhouette_scores)


# ====================================================================== #
# 7) Évaluation des clusters
# ====================================================================== #

metric_combined = show_metric(labels_true, labels_pred, embeddings_norm,bool_show=True, name_descriptor="COMBINED", bool_return=True)

print("- Exporting data for visualization")
df_metric = pd.DataFrame([metric])
# Convert to 3D for visualization
x_3d = conversion_3d(embeddings_norm)
df_export = create_df_to_export(x_3d, labels_true, labels_pred)
df_export["image"] = paths
print(df_export["image"])
# Ensure output directory exists
os.makedirs(PATH_OUTPUT, exist_ok=True)
# Save results
df_export.to_excel(PATH_OUTPUT + "/save_clustering_SIM.xlsx")
df_metric.to_excel(PATH_OUTPUT + "/save_metric_SIM.xlsx")
print("Done! To visualize results, run: streamlit run dashboard_clustering_mixed.py")
# Save the silhouette and inertia scores
np.save(os.path.join(PATH_OUTPUT, "silhouette_scores_SIM.npy"), silhouette_scores)
np.save(os.path.join(PATH_OUTPUT, "inertie_scores_SIM.npy"), inertie_scores)