import torchvision.transforms as transforms
import torch
from torchvision import models
from skimage.feature import hog
from sklearn.cluster import KMeans as KMeansSL
import numpy as np
import cv2
import os
import clip
from PIL import Image

####################################
#image preparation
####################################

def resize(image, target_size=(224, 224)):
    """Resizes an image in the array to the target size using OpenCV."""
    resized_img = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return resized_img


def crop_center(image):
    """Crops an image to a square by taking the center part."""
    height, width = image.shape[:2]
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    cropped_img = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cropped_img


def blur_image(image, kernel_size=5):
    """Apply Gaussian blur to an image to reduce noise."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)



def padding(image):
    """Adds padding to each image to make it a square."""
    height, width = image.shape[:2]
    max_dim = max(height, width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    # Pad with zeros (black)
    padded_img = cv2.copyMakeBorder(image, pad_height, max_dim - height - pad_height,
                                        pad_width, max_dim - width - pad_width,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return np.array(image)



def shuffle_images_and_labels(images_array, labels_array,features_array,paths_array):
    """Shuffles the images and labels arrays and features array and paths array while maintaining their correspondence. """
    indices = np.arange(len(images_array))
    np.random.shuffle(indices)
    return images_array[indices], labels_array[indices],features_array[indices], paths_array[indices]


#The image has strong color casts
def normalize_color(img):
    """Enhanced color normalization with white balancing"""
    # Convert to LAB and split channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    
    # White balancing (ensure same dtype)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    avg_a, avg_b = np.mean(a), np.mean(b)
    a_balanced = (a - (avg_a - 128)).clip(0, 255).astype(np.uint8)
    b_balanced = (b - (avg_b - 128)).clip(0, 255).astype(np.uint8)
    
    # Merge and convert back to BGR
    lab_norm = cv2.merge((l_norm, a_balanced, b_balanced))
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)






####################################
#feature extraction
####################################



#################HOG needs resize#################
def extract_hog_features(image):
    """extracts hog features for an image. it converts first image to grey scale"""
    if len(image.shape) == 2:gray = image
    else: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, visualize=True)
    return features

def extract_hog_features_batch(images):
    return [extract_hog_features(image) for image in images]


#################HOG doesn't need resize#################
def extract_color_histogram(image, bins=(8, 8, 8)):
    """extracts histogram features from an image."""
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_color_histogram_batch(images, bins=(8, 8, 8)):
    return [extract_color_histogram(image, bins) for image in images]


#################Color Stats doesn't need resize#################
def extract_color_stats(image):
    """extracts color stats features from an image."""
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    return np.concatenate([means, stds])

def extract_color_stats_batch(images):
    return [extract_color_stats(image) for image in images]

#################SIFT doesn't need resize#################
sift = cv2.SIFT_create()
def extract_sift_features(image):
    """extracts sift features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors 

def extract_sift_features_batch(images):
    return [extract_sift_features(image) for image in images]



def compute_sift_representations(images, num_clusters=100):
    """
    Computes two representations for SIFT features:
    1. Mean descriptor (fixed-size 128 vector per image)
    2. Bag of Visual Words (BoVW histogram per image)
    """
    sift_descriptors = extract_sift_features_batch(images)

    mean_features = []
    for desc in sift_descriptors:
        if desc is not None: mean_descriptor = np.mean(desc, axis=0)
        else: mean_descriptor = np.zeros(128)
        mean_features.append(mean_descriptor)
    
    #mean_features = np.array(mean_features)  # Shape: (num_images, 128)

    # === BoVW Method ===
    # Flatten all descriptors into one array for clustering
    all_descriptors = np.vstack(sift_descriptors)  # Shape: (total_keypoints, 128)
    kmeans = KMeansSL(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    bow_features = []
    for desc in sift_descriptors:
        labels = kmeans.predict(desc)
        hist, _ = np.histogram(labels, bins=np.arange(num_clusters+1))
        bow_features.append(hist)

    bow_features = bow_features 
    return mean_features, bow_features






#################Contour doesn't need resize#################
def extract_contour_features(image):
    """extracts contour features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
    return [np.mean(areas) if areas else 0, np.mean(perimeters) if perimeters else 0]

def extract_contour_features_batch(images):
    return [extract_contour_features(image) for image in images]



#################ResNet50 needs resize#################

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load Pretrained ResNet50 (Backbone for DeepCluster)
resnet_model = models.resnet50(weights="IMAGENET1K_V2")
resnet_model.fc = torch.nn.Identity()  # Remove last layer
# Set model to evaluation mode
resnet_model.eval()

def extract_deepcluster_features(image):
    """Extract CNN features using ResNet50."""
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(image)
    return features.numpy().flatten()

def extract_deepcluster_features_batch(images):
    return [extract_deepcluster_features(image) for image in images]



#################CLIP needs resize#################

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device) 

def extract_clip_features(image):
    """extracts CLIP (Contrastive Language–Image Pretraining) (CNN) features from an image. first if the image is a numpy array we convert it to a PIL array"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(): features = clip_model.encode_image(image)
    return features.cpu().detach().numpy().flatten()

def extract_clip_features_batch(images):
    return [extract_clip_features(image) for image in images]





####################################
#image loading
####################################
def load_images_and_labels(data_dir,withPaths=False):
    images = []  # List to store all images
    labels = []  # List to store labels for each image
    paths = []
    
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
                image = cv2.resize(image, (224, 224))
                #image = blur_image(image)
                # Append the image and its corresponding label
                images.append(image)
                labels.append(label)  # label is 0 for apple, 1 for banana, 2 for candy

                if withPaths :  paths.append(image_path)
    
    # Convert images and labels to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    if withPaths : return images_array, labels_array, np.array(paths)
    return images_array, labels_array







if __name__ == "__main__":
    img_path = "../data/val/pretzel/03ab7e84c8a781ed.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Image not loaded")
    else:
        # Normalize and show
        normalized_img = normalize_color(img)
        cv2.imshow('image',normalized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 