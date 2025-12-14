import io
import base64
import time
import numpy as np
import random
import os
import pandas as pd
from numpy import linalg as la
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import fetch_olivetti_faces, load_digits

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE ---
STATE = {
    "data": None,          # The training dataset (N, Features)
    "shape": (64, 64),     # Original image shape
    "centroids": None,     # Current K-Means centroids
    "labels": None,        # Current cluster labels
    "k_current": 0,        # To track if we need to re-train
    "algo_current": ""     # To track if algorithm changed
}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Setup
source_csv = "assets/forest/covtype.csv"
save_folder = "test_csv_samples"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print(f"Reading {source_csv}...")

# 2. Read the Data
# We only need a few rows
df = pd.read_csv(source_csv, nrows=20) 

# Drop label/target columns if they exist
cols_to_drop = [c for c in df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# 3. Normalize (Crucial!)
# The web app does this internally, so we must do it here too 
# to make the "image" look right.
data = df.values.astype(float)
min_val = data.min(axis=0)
max_val = data.max(axis=0)
data = (data - min_val) / (max_val - min_val + 1e-8)

# 4. Save as Images
print(f"Saving 5 samples to '{save_folder}'...")

for i in range(5):
    # Get one row (one forest area)
    row_vector = data[i]
    
    # Reshape: 1 pixel tall, N pixels wide (Barcode)
    # or make it square-ish for visibility if you prefer
    n_features = len(row_vector)
    
    # Let's repeat the rows so the image is bigger and easier to see
    # We will make it 50 pixels tall, but the pattern is only 1 pixel wide vertical
    img_display = np.tile(row_vector, (50, 1))
    
    filename = f"{save_folder}/forest_sample_{i}.png"
    
    # Save using grayscale colormap
    plt.imsave(filename, img_display, cmap='gray')
    print(f"  Saved: {filename}")

print("\n✅ Done! Upload these files to your web app.")

# --- YOUR ALGORITHMS ---

def k_means_first_var(data, k):
    """ Variant 1: Random Clusters Initialization """
    m, n = data.shape
    idx = np.arange(k)
    clusters = random.choices(idx, k=m)
    clusters = np.array(clusters)
    
    c_old = np.zeros([k, n]) + 9999
    c = np.zeros([k, n])
    
    # Initialize centroids to avoid empty cluster errors on first run
    for j in range(k):
        points = data[clusters == j]
        if len(points) > 0:
            c[j] = np.mean(points, axis=0)
        else:
            c[j] = data[np.random.randint(m)]

    while (not np.allclose(c, c_old, atol=1e-4)):
        c_old = c.copy()
        
        # 1. Update Centroids (M-Step)
        for j in range(k):
            points_in_cluster = data[clusters == j]
            if len(points_in_cluster) > 0:
                c[j] = np.mean(points_in_cluster, axis=0)
            else:
                c[j] = data[np.random.randint(m)]
        
        # 2. Update Clusters (E-Step)
        # Vectorized distance for speed: dist[i,j] = norm(data[i] - c[j])
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
        
        # 2. Update Centroids
        for j in range(k):
            points_in_cluster = data[clusters == j]
            if len(points_in_cluster) > 0:
                c[j] = np.mean(points_in_cluster, axis=0)
            else:
                c[j] = data[np.random.randint(m)]
                
    return clusters, c

# --- HELPERS ---
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
        # --- Built-in ---
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

        # --- Custom Logic (Handles Images AND nested CSVs) ---
        else:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(backend_dir, '..', 'assets', name)
            
            # Detect if user pointed to a CSV directly or a Folder containing a CSV
            target_csv = None
            
            # Check 1: Is it 'assets/forest.csv'?
            if os.path.exists(base_path + ".csv"):
                target_csv = base_path + ".csv"
            
            # Check 2: Is it 'assets/forest/' folder containing a CSV?
            elif os.path.isdir(base_path):
                files_in_dir = os.listdir(base_path)
                csvs = [f for f in files_in_dir if f.lower().endswith('.csv')]
                if len(csvs) > 0:
                    target_csv = os.path.join(base_path, csvs[0]) # Pick the first CSV found

            # --- BRANCH A: Load CSV Data ---
            if target_csv:
                print(f"Reading CSV: {target_csv}")
                # Limit to 5000 rows for speed (Covertype is HUGE)
                df = pd.read_csv(target_csv)
                
                # Sampling if too big
                if len(df) > 5000:
                    print("Dataset too large, sampling 5000 rows...")
                    df = df.sample(n=5000, random_state=42)

                # Remove labels/targets
                cols_to_drop = [c for c in df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                
                # Convert to Matrix
                data_matrix = df.values.astype(float)
                
                # Normalize columns to 0-1 range (Critical for K-Means on tabular data)
                # (axis=0 means normalize per column/feature)
                min_val = data_matrix.min(axis=0)
                max_val = data_matrix.max(axis=0)
                # Avoid division by zero
                data_matrix = (data_matrix - min_val) / (max_val - min_val + 1e-8)
                
                STATE['data'] = data_matrix
                
                # Auto-Detect Shape for Visualization
                n_features = data_matrix.shape[1]
                side = int(np.sqrt(n_features))
                if side * side == n_features:
                    STATE['shape'] = (side, side)
                else:
                    # Forest Covertype has 54 columns -> Not square -> Render as Barcode
                    STATE['shape'] = (1, n_features)
                    
                STATE['name'] = name

            # --- BRANCH B: Load Images from Folder ---
            elif os.path.isdir(base_path):
                print(f"Scanning folder for images: {base_path}")
                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm') 
                image_paths = []
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith(valid_exts):
                            image_paths.append(os.path.join(root, file))
                
                if not image_paths: return jsonify({"error": "No images or CSV found"}), 400

                target_size = (100, 100)
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
                return jsonify({"error": "Dataset path not found"}), 404

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
    
    # 1. Train K-Means if params changed or not trained yet
    if STATE['k_current'] != k or STATE['algo_current'] != algo or STATE['centroids'] is None:
        print(f"Training K-Means ({algo}, k={k})...")
        
        # Create a temp variable for labels
        new_labels = None
        new_centroids = None

        if algo == 'var1':
            new_labels, new_centroids = k_means_first_var(STATE['data'], k)
        else:
            new_labels, new_centroids = k_means_second_var(STATE['data'], k)
            
        # Update Global State properly
        STATE['centroids'] = new_centroids
        STATE['labels'] = new_labels  # <--- THIS WAS MISSING
        STATE['k_current'] = k
        STATE['algo_current'] = algo
        
    # 2. Process Input Image
    img = Image.open(file).convert('L')
    img = img.resize((STATE['shape'][1], STATE['shape'][0]))
    vec = np.array(img).flatten().astype(float) / 255.0
    
    # 3. Find Nearest Cluster
    dists = la.norm(STATE['centroids'] - vec, axis=1)
    cluster_id = np.argmin(dists)
    
    # --- ADD THIS DEBUG PRINT ---
    # Count how many items are in the chosen cluster
    points_in_cluster = np.sum(STATE['labels'] == cluster_id)
    print(f"⚠️ Selected Cluster #{cluster_id} contains {points_in_cluster} images.")
    # ----------------------------
    
    # 4. Return Data
    # centroid image = The centroid (average) of the cluster
    centroid_b64 = array_to_b64(STATE['centroids'][cluster_id], STATE['shape'])
    input_b64 = array_to_b64(vec, STATE['shape'])
    
    return jsonify({
        "image_b64": input_b64,
        "centroid_b64": centroid_b64,
        "algorithm": f"K-Means ({algo})",
        "person_label": f"Cluster #{cluster_id}",
        "nearest_idx": int(cluster_id)
    })

@app.route('/run_statistics', methods=['POST'])
def stats():
    if STATE['data'] is None: return jsonify({"error": "No data"}), 400
    
    var1_res = []
    var2_res = []
    k_vals = [2, 3, 5, 8]
    
    data = STATE['data']
    
    for k in k_vals:
        # Test Var 1
        t0 = time.time()
        k_means_first_var(data, k)
        t1 = time.time()
        var1_res.append({"name": f"Var1 (K={k})", "accuracy": 0, "time_ms": (t1-t0)*1000})
        
        # Test Var 2
        t0 = time.time()
        k_means_second_var(data, k)
        t1 = time.time()
        var2_res.append({"name": f"Var2 (K={k})", "accuracy": 0, "time_ms": (t1-t0)*1000})
        
    return jsonify({"classification": var1_res, "preprocessing": var2_res})

if __name__ == '__main__':
    app.run(port=5000, debug=True)