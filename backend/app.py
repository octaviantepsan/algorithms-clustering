import io
import base64
import time
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from numpy import linalg as la
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.metrics import silhouette_score
from PIL import Image, ImageDraw

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE ---
STATE = {
    "data": None,          # The training dataset (N, Features)
    "shape": (64, 64),     # Original image shape
    "ghosts": None,     # Current K-Means ghosts
    "labels": None,        # Current cluster labels
    "k_current": 0,        # To track if we need to re-train
    "algo_current": ""     # To track if algorithm changed
}

# 1. Clean old data to prevent mixing
save_folder = "assets/dots"
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder)

print("Generating Clean Blobs...")

# --- CLUSTER 1: TOP-LEFT ---
# Tight Group: Center is (15,15), Jitter is only +/- 2 pixels
for i in range(50):
    # Create pure black 64x64 image (Mode 'L' = 8-bit Grayscale)
    img = Image.new('L', (64, 64), color=0)
    draw = ImageDraw.Draw(img)
    
    # Random position VERY close to 15,15
    r = np.random.randint(13, 18) 
    c = np.random.randint(13, 18)
    
    # Draw a 10x10 White Square (255)
    # The large size + small jitter guarantees overlap -> High Silhouette
    draw.rectangle([c, r, c+10, r+10], fill=255)
    
    img.save(f"{save_folder}/dot_tl_{i}.png")

# --- CLUSTER 2: BOTTOM-RIGHT ---
# Tight Group: Center is (45,45), Jitter is only +/- 2 pixels
for i in range(50):
    img = Image.new('L', (64, 64), color=0)
    draw = ImageDraw.Draw(img)
    
    r = np.random.randint(43, 48)
    c = np.random.randint(43, 48)
    
    draw.rectangle([c, r, c+10, r+10], fill=255)
    
    img.save(f"{save_folder}/dot_br_{i}.png")

print("✅ OPTIMIZED dataset created in assets/dots")

# --- YOUR ALGORITHMS ---

def k_means_first_var(data, k):
    """ Variant 1: Random Clusters Initialization """
    m, n = data.shape
    idx = np.arange(k)
    clusters = random.choices(idx, k=m)
    clusters = np.array(clusters)
    
    c_old = np.zeros([k, n]) + 9999
    c = np.zeros([k, n])
    
    # Initialize ghosts
    for j in range(k):
        points = data[clusters == j]
        if len(points) > 0:
            c[j] = np.mean(points, axis=0)
        else:
            c[j] = data[np.random.randint(m)]

    while (not np.allclose(c, c_old, atol=1e-4)):
        c_old = c.copy()
        
        # 1. Update ghosts (M-Step)
        for j in range(k):
            points_in_cluster = data[clusters == j]
            if len(points_in_cluster) > 0:
                c[j] = np.mean(points_in_cluster, axis=0)
            else:
                c[j] = data[np.random.randint(m)]
        
        # 2. Update Clusters (E-Step)
        distances = np.zeros([m, k])
        for j in range(k):
            distances[:, j] = la.norm(data - c[j], axis=1)
            
        clusters = np.argmin(distances, axis=1)
            
    return clusters, c

def k_means_second_var(data, k):
    """ Variant 2: Random Points Initialization (Forgy) """
    m, n = data.shape
    random_idx = np.random.permutation(m)
    c = data[random_idx[:k], :].copy()
    
    c_old = np.zeros([k, n]) + 9999
    clusters = np.zeros(m)
    
    while (not np.allclose(c, c_old, atol=1e-4)):
        c_old = c.copy()
        
        # 1. Update Clusters
        distances = np.zeros([m, k])
        for j in range(k):
            distances[:, j] = la.norm(data - c[j], axis=1)    
        clusters = np.argmin(distances, axis=1)
        
        # 2. Update ghosts
        for j in range(k):
            points_in_cluster = data[clusters == j]
            if len(points_in_cluster) > 0:
                c[j] = np.mean(points_in_cluster, axis=0)
            else:
                c[j] = data[np.random.randint(m)]
                
    return clusters, c

# --- HELPERS ---
def calculate_inertia(data, centroids, labels):
    """
    Calculates the Sum of Squared Errors (Inertia).
    Formula: Sum of distance(point, assigned_centroid)^2
    """
    inertia = 0
    # For each cluster
    for k_idx, centroid in enumerate(centroids):
        # Get all points belonging to this cluster
        cluster_points = data[labels == k_idx]
        if len(cluster_points) > 0:
            # Calculate squared Euclidean distance for these points
            diff = cluster_points - centroid
            sq_dist = np.sum(diff**2)
            inertia += sq_dist
    return inertia

def array_to_b64(arr, shape):
    try:
        arr = arr.reshape(shape)
        # Normalize to 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8)).convert('L')
        buff = io.BytesIO()
        img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(e)
        return ""

# --- ROUTES ---

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    req = request.json
    name = req.get('dataset', 'attfaces')
    global STATE
    
    print(f"Loading {name}...")
    try:
        # --- Built-in Datasets ---
        if name == 'digits':
            data = load_digits()
            STATE['data'] = data.data
            STATE['shape'] = (8, 8)
            STATE['name'] = 'digits'
            
        elif name == 'attfaces':
            data = fetch_olivetti_faces()
            STATE['data'] = data.data
            STATE['shape'] = (64, 64)
            STATE['name'] = 'attfaces'

        # --- Custom Folders (Generic) ---
        else:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(backend_dir, '..', 'assets', name)
            
            # This check finds 'dots', 'gems', 'trains', etc. automatically!
            if os.path.isdir(base_path):
                print(f"Scanning folder: {base_path}")
                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm') 
                image_paths = []
                
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith(valid_exts):
                            image_paths.append(os.path.join(root, file))
                
                if not image_paths: 
                    return jsonify({"error": f"No images found in assets/{name}"}), 400

                target_size = (100, 100) # Standardize size
                image_list = []

                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize((target_size[1], target_size[0]))
                        vec = np.array(img).flatten().astype(float) / 255.0
                        image_list.append(vec)
                    except: pass
                
                STATE['data'] = np.array(image_list)
                STATE['shape'] = target_size
                STATE['name'] = name
            
            else:
                return jsonify({"error": f"Folder '../assets/{name}' not found"}), 404

        # Reset Training
        STATE['centroids'] = None
        STATE['labels'] = None
        STATE['k_current'] = 0
        
        return jsonify({"msg": f"Loaded {name} successfully", "shape": STATE['shape']}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return jsonify({"image_b64": base64.b64encode(buff.getvalue()).decode("utf-8")})

@app.route('/process_image', methods=['POST'])
def process():
    global STATE
    if STATE['data'] is None:
        return jsonify({"error": "Load dataset first"}), 400
        
    file = request.files['image']
    algo = request.form.get('algorithm', 'var1')
    k = int(request.form.get('k', 3))
    
    # 1. Train K-Means if needed
    if STATE['k_current'] != k or STATE['algo_current'] != algo or STATE['ghosts'] is None:
        print(f"Training K-Means ({algo}, k={k})...")
        
        new_labels = None
        new_ghosts = None

        if algo == 'var1':
            new_labels, new_ghosts = k_means_first_var(STATE['data'], k)
        else:
            new_labels, new_ghosts = k_means_second_var(STATE['data'], k)
            
        STATE['ghosts'] = new_ghosts
        STATE['labels'] = new_labels
        STATE['k_current'] = k
        STATE['algo_current'] = algo
        
    # 2. Process Input Image
    img = Image.open(file).convert('L')
    img = img.resize((STATE['shape'][1], STATE['shape'][0]))
    vec = np.array(img).flatten().astype(float) / 255.0
    
    # 3. Find Nearest Cluster
    dists = la.norm(STATE['ghosts'] - vec, axis=1)
    cluster_id = np.argmin(dists)
    
    # Debug info
    points_in_cluster = np.sum(STATE['labels'] == cluster_id)
    print(f"⚠️ Selected Cluster #{cluster_id} contains {points_in_cluster} images.")
    
    # 4. Return Data
    # FIX: Changed 'ghost_b64' to 'ghost_b64' to match your Frontend
    ghost_b64 = array_to_b64(STATE['ghosts'][cluster_id], STATE['shape'])
    input_b64 = array_to_b64(vec, STATE['shape'])
    
    return jsonify({
        "image_b64": input_b64,
        "ghost_b64": ghost_b64,  # <--- FIXED NAME
        "algorithm": f"K-Means ({algo})",
        "person_label": f"Cluster #{cluster_id}",
        "nearest_idx": int(cluster_id)
    })

@app.route('/run_statistics', methods=['POST'])
def stats():
    if STATE['data'] is None: 
        return jsonify({"error": "No dataset loaded. Please load a dataset first."}), 400
    
    data = STATE['data']
    # Limit data for statistics to speed it up (optional, removes lag on huge datasets)
    # if len(data) > 1000: data = data[:1000]

    results_var1 = []
    results_var2 = []
    
    # We test K from 2 to 10 (Elbow Method usually looks at this range)
    k_range = range(2, 11) 
    
    print("Starting Statistics Loop...")

    for k in k_range:
        # --- ALGORITHM 1 (Random Clusters) ---
        t0 = time.time()
        labels1, centroids1 = k_means_first_var(data, k)
        t1 = time.time()
        
        inertia1 = calculate_inertia(data, centroids1, labels1)
        # Silhouette requires at least 2 labels. If k=1 or all points in 1 cluster, it fails.
        sil1 = -1 
        if len(np.unique(labels1)) > 1:
            sil1 = silhouette_score(data, labels1)
            
        results_var1.append({
            "k": k,
            "time": (t1 - t0),       # Seconds
            "inertia": inertia1,     # For Elbow Method
            "silhouette": sil1       # For Quality Check
        })

        # --- ALGORITHM 2 (Random Points) ---
        t0 = time.time()
        labels2, centroids2 = k_means_second_var(data, k)
        t1 = time.time()
        
        inertia2 = calculate_inertia(data, centroids2, labels2)
        sil2 = -1
        if len(np.unique(labels2)) > 1:
            sil2 = silhouette_score(data, labels2)

        results_var2.append({
            "k": k,
            "time": (t1 - t0),
            "inertia": inertia2,
            "silhouette": sil2
        })
        
        print(f"Computed stats for K={k}")

    return jsonify({
        "var1": results_var1, 
        "var2": results_var2
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)