import tensorflow as tf
import numpy as np
import random
from constant import PATH_OUTPUT, PATH_DATA, PATH_DATA2,PATH_DATA3, MODEL_CLUSTERING
from features import *

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K

# Pour PCA et KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from clustering import *

import os

# ====================================================================== #
# 1) Chargement des images
# ====================================================================== #

BATCH_SIZE = 64 #chaque batch va passer un à un dans le modèle pour l'entrainement, plus le batch est grand, moins il y a d'overfitting
IMG_HEIGHT = 128
IMG_WIDTH = 128

images_array, labels_true = load_images_and_labels(PATH_DATA2)
rng = np.random.default_rng(seed=42)
images_array_suffled = rng.permutation(images_array)
# images_array.shape => (N, 128, 128, 3)

# ====================================================================== #
# 2) Préparation du dataset + augmentations aléatoires (SimCLR)
# ====================================================================== #

def random_augmentation(image): 
    """Applique des transformations aléatoires de type augmentation SimCLR."""
    #d'après https://proceedings.mlr.press/v119/chen20j/chen20j.pdf crop et color sont les transfos qui donnent la meilleur accuracy

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    # flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # crop
    image = tf.image.random_crop(tf.image.pad_to_bounding_box(image, 4, 4, IMG_HEIGHT+8, IMG_WIDTH+8), [IMG_HEIGHT, IMG_WIDTH, 3])
    # brightness/contrast
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=2.0)
    return image

def prepare_contrastive_batch(img):
    """
    Crée 2 vues augmentées à partir d'une même image.
    Retourne (vue1, vue2).
    """
    #ces 2 vues vont passer dans le modèle, les paramètres du modèle vont être modifiés de sorte à 
    # maximiser la similarité entre ces 2 vues
    # minimser la similarité entre chaqune de ces 2 vues et les 2N - 2 autres vues du batch -> problème de faux négatifs
    img1 = random_augmentation(img)
    img2 = random_augmentation(img)
    return (img1, img2)

# Conversion en dataset de taille N = BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices(images_array_suffled) 
dataset = dataset.map(prepare_contrastive_batch, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ====================================================================== #
# 3) Définition du modèle SimCLR
# ====================================================================== #

PROJ_DIM = 128  
BASE_EMB_DIM = 2048  # sortie ResNet50 après GAP

def create_encoder():
    """
    Encodeur ResNet50 tronqué qui renvoie des features (2048D).
    """
    base_model = ResNet50(weights='imagenet', include_top=False,input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Optionnel: freeze partiel si nécessaire
    # for layer in base_model.layers[:-10]:
    #     layer.trainable = False

    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inputs, x, name="resnet50_encoder") 


def create_projection_head():
    """
    Projette les features 2048D en un espace latent (128D).
    """
    inputs = layers.Input((BASE_EMB_DIM,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(PROJ_DIM)(x)  # Pas d'activation
    return models.Model(inputs, x, name="projection_head")


class SimCLR(tf.keras.Model):
    """
    Modèle global = Encoder + Projection Head (SimCLR).
    """
    def __init__(self):
        super().__init__()
        self.encoder = create_encoder()
        self.projection_head = create_projection_head()
    
    def call(self, x, training=False):
        h = self.encoder(x, training=training)       # (batch, 2048)
        z = self.projection_head(h, training=training)  # (batch, 128)
        z = tf.math.l2_normalize(z, axis=1)          # normalisation L2
        return z

def simclr_loss(z1, z2, temperature=0.1):
    """
    Calcule la perte InfoNCE pour un batch:
      z1, z2: (batch_size, proj_dim)
    """
    batch_size = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)  # (2N, proj_dim)
    
    sim_matrix = tf.matmul(z, z, transpose_b=True)  # (2N, 2N)
    sim_matrix = sim_matrix / temperature
    
    # Masque diagonale
    mask = tf.eye(2 * batch_size)
    logits = sim_matrix * (1 - mask)  # supprime la diagonale i==i
    
    # Paires positives: i+N (mod 2N)
    positives = tf.concat([tf.range(batch_size, 2*batch_size),
                           tf.range(0, batch_size)], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=positives, logits=logits
    )
    loss = tf.reduce_mean(loss)
    return loss

# ====================================================================== #
# 4) Entraînement SimCLR
# ====================================================================== #

model = SimCLR()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9) moins performant que Adam



TEMPERATURE = 0.1
EPOCHS = 20
BATCH_SIZE_EMB = 64 #a la fin, les 4000 images sont passées batch par batch, ca n'a pas d'impact sur le modèle, c'est juste pour la mémoire


for epoch in range(EPOCHS):
    epoch_loss = []
    for step, (vues1, vues2) in enumerate(dataset):
        with tf.GradientTape() as tape:
            z1 = model(vues1, training=True)
            z2 = model(vues2, training=True)
            loss_value = simclr_loss(z1, z2, temperature=TEMPERATURE)
        
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss.append(loss_value.numpy())
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {np.mean(epoch_loss):.4f}")

# ====================================================================== #
# 5) Extraction des embeddings (encoder seul)
# ====================================================================== #

encoder = model.encoder

all_embeddings = []

ds_inference = tf.data.Dataset.from_tensor_slices(images_array).batch(BATCH_SIZE_EMB)

for batch_imgs in ds_inference:
    feats = encoder(batch_imgs, training=False)  # shape: (batch, 2048) 
    all_embeddings.append(feats.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 2048)

np.save(os.path.join(PATH_OUTPUT, "SimCLRFinal.npy"), all_embeddings)
print("Embeddings sauvegardés avec succès.")