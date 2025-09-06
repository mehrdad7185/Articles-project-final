from flask import Flask, jsonify, request
import time
import docker
import threading
import os
import numpy as np
import tensorflow as tf
import joblib
from collections import deque

app = Flask(__name__)

# --- Configuration ---
FOG_NODES = ['fog-node-1', 'fog-node-2']
MODEL_PATH = 'digital_twin_model.h5'
SCALER_PATH = 'scaler.joblib'
SEQUENCE_LENGTH = 10  # Must be the same as used in training
PREDICTION_THRESHOLD = 0.7 # If prediction > 70%, the node is considered at-risk

# --- State Management ---
status_lock = threading.Lock()
# Store current metrics for each node
node_status = {node: {"status": "UP", "cpu": 0.0, "memory": 0.0, "latency": 0.0} for node in FOG_NODES}
# Store a history (sequence) of recent metrics for prediction
node_history = {node: deque(maxlen=SEQUENCE_LENGTH) for node in FOG_NODES}

# --- Model and Docker Client Initialization ---
model = None
scaler = None
DOCKER_AVAILABLE = False

def load_dependencies():
    """Load the ML model, scaler, and connect to Docker."""
    global model, scaler, DOCKER_AVAILABLE, client
    
    # Load the trained Digital Twin model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print(f"[INFO] Digital Twin model ('{MODEL_PATH}') and scaler loaded successfully.")
        except Exception as e:
            print(f"[CRITICAL_ERROR] Could not load ML model or scaler: {e}")
            model = None # Ensure model is None if loading fails
    else:
        print("[WARNING] Model or scaler file not found. Scheduler will run in REACTIVE mode only.")

    # Connect to Docker daemon
    try:
        client = docker.from_env()
        DOCKER_AVAILABLE = True
        print("[INFO] Successfully connected to Docker daemon.")
    except Exception as e:
        DOCKER_AVAILABLE = False
        print(f"[CRITICAL_ERROR] Could not connect to Docker daemon: {e}")

def predict_failure_risk(node_name):
    """Predicts the failure risk for a node based on its recent history."""
    if model is None or scaler is None:
        return 0.0 # Cannot predict if model/scaler isn't loaded

    with status_lock:
        history = list(node_history[node_name])

    # We can only predict if we have a full sequence of data
    if len(history) < SEQUENCE_LENGTH:
        return 0.0

    # Prepare the data for the model
    sequence = np.array([list(item.values()) for item in history])
    scaled_sequence = scaler.transform(sequence)
    reshaped_sequence = np.reshape(scaled_sequence, (1, SEQUENCE_LENGTH, scaled_sequence.shape[1]))

    # Make the prediction
    try:
        prediction = model.predict(reshaped_sequence, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"[ERROR] Prediction failed for node {node_name}: {e}")
        return 0.0

def update_node_metrics():
    """Periodically fetches metrics for all nodes and updates their history."""
    while True:
        if not DOCKER_AVAILABLE:
            time.sleep(10)
            continue
        
        for node_name in FOG_NODES:
            try:
                container = client.containers.get(node_name)
                container.reload() # Refresh container data

                if container.status == 'running':
                    stats = container.stats(stream=False)
                    # --- CPU Calculation ---
                    cpu_percent = 0.0
                    precpu_stats = stats.get("precpu_stats", {})
                    cpu_stats = stats.get("cpu_stats", {})
                    if precpu_stats and 'cpu_usage' in precpu_stats and 'system_cpu_usage' in precpu_stats:
                        cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
                        system_cpu_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
                        online_cpus = cpu_stats.get("online_cpus", 1)
                        if system_cpu_delta > 0.0 and cpu_delta > 0.0:
                            cpu_percent = (cpu_delta / system_cpu_delta) * online_cpus * 100.0
                    
                    # --- Memory Calculation ---
                    memory_usage = stats.get("memory_stats", {}).get("usage", 0) / (1024 * 1024)

                    with status_lock:
                        # Update the current status
                        node_status[node_name].update({"status": "UP", "cpu": round(cpu_percent, 2), "memory": round(memory_usage, 2)})
                        # Append the latest metrics to the history deque for prediction
                        # Note: We use a placeholder for latency as it's not available from Docker stats.
                        # This should be updated by a real-world metric if available. For now, it's 0.
                        node_history[node_name].append({'cpu': cpu_percent, 'memory': memory_usage, 'latency': 0.0})

                else: # If container is not running
                    with status_lock:
                        node_status[node_name]["status"] = "DOWN"
                        node_status[node_name]["cpu"] = 0.0
                        node_status[node_name]["memory"] = 0.0
                        node_history[node_name].clear() # Clear history for down nodes

            except docker.errors.NotFound:
                 with status_lock:
                    node_status[node_name]["status"] = "DOWN"
            except Exception as e:
                print(f"[ERROR] Could not get stats for {node_name}: {e}")
        
        time.sleep(2) # Update metrics every 2 seconds

@app.route('/get_fog_node', methods=['GET'])
def get_fog_node():
    with status_lock:
        # Get a list of nodes that are currently running
        active_nodes = {node: info for node, info in node_status.items() if info["status"] == "UP"}
    
    if not active_nodes:
        return jsonify({"error": "No active fog nodes available"}), 503

    # --- Proactive Decision Logic ---
    node_evaluations = []
    for node_name, stats in active_nodes.items():
        risk_score = predict_failure_risk(node_name)
        node_evaluations.append({
            "name": node_name,
            "cpu": stats['cpu'],
            "risk": risk_score
        })

    # Filter out nodes that are predicted to be at high risk
    safe_nodes = [n for n in node_evaluations if n['risk'] < PREDICTION_THRESHOLD]
    
    print(f"STATUS_UPDATE::{node_evaluations}") # Log the full evaluation

    if safe_nodes:
        # If there are safe nodes, choose the one with the lowest CPU from that pool
        chosen_node = min(safe_nodes, key=lambda node: node['cpu'])
        print(f"[PROACTIVE] Chosen safe node: {chosen_node['name']} (CPU: {chosen_node['cpu']:.2f}%, Risk: {chosen_node['risk']:.2f})")
    else:
        # Fallback: If all nodes are at risk, choose the one with the lowest risk score
        chosen_node = min(node_evaluations, key=lambda node: node['risk'])
        print(f"[FALLBACK] All nodes at risk. Chosen node with lowest risk: {chosen_node['name']} (CPU: {chosen_node['cpu']:.2f}%, Risk: {chosen_node['risk']:.2f})")

    return jsonify({"fog_node_host": chosen_node['name']})

if __name__ == '__main__':
    load_dependencies()
    # Start the background thread for continuous metric collection
    metric_thread = threading.Thread(target=update_node_metrics, daemon=True)
    metric_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
