# scheduler_reactive.py (Your advanced baseline from GitHub)

from flask import Flask, jsonify, request
import time
import docker
import threading

app = Flask(__name__)

# --- Configuration ---
FOG_NODES = ['fog-node-1', 'fog-node-2']
RECOVERY_TIME_SECONDS = 30

# --- State Management ---
status_lock = threading.Lock()
node_status = {
    node: {"status": "UP", "last_failure": 0, "cpu": 0.0, "memory": 0.0}
    for node in FOG_NODES
}

# --- Docker Client Initialization ---
try:
    client = docker.from_env()
    DOCKER_AVAILABLE = True
    print("[INFO] Successfully connected to Docker daemon.")
except Exception as e:
    DOCKER_AVAILABLE = False
    print(f"[CRITICAL_ERROR] Could not connect to Docker daemon: {e}")

def mark_node_down(node_name, reason):
    """A centralized function to mark a node as down and reset its metrics."""
    with status_lock:
        if node_name in node_status and node_status[node_name]['status'] == 'UP':
            print(f"[HEALTH CHECK] Node '{node_name}' FAILED ({reason}) and is now SUSPECTED.")
            node_status[node_name]['status'] = 'SUSPECTED'
            node_status[node_name]['last_failure'] = time.time()
            node_status[node_name]['cpu'] = 0.0
            node_status[node_name]['memory'] = 0.0

def update_all_node_statuses():
    if not DOCKER_AVAILABLE: return

    for node_name in FOG_NODES:
        try:
            container = client.containers.get(node_name)
            container.reload()

            if container.status == 'running':
                with status_lock:
                    if node_status[node_name]["status"] == "SUSPECTED":
                        print(f"[HEALTH CHECK] Node '{node_name}' has RECOVERED and is now UP.")
                        node_status[node_name]["status"] = "UP"
                
                stats = container.stats(stream=False)
                cpu_percent = 0.0
                precpu_stats = stats.get("precpu_stats", {})
                cpu_stats = stats.get("cpu_stats", {})
                if precpu_stats and 'cpu_usage' in precpu_stats and 'system_cpu_usage' in precpu_stats:
                    cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
                    system_cpu_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
                    online_cpus = cpu_stats.get("online_cpus", 1)
                    if system_cpu_delta > 0.0 and cpu_delta > 0.0:
                        cpu_percent = (cpu_delta / system_cpu_delta) * online_cpus * 100.0
                memory_usage = stats.get("memory_stats", {}).get("usage", 0) / (1024 * 1024)
                
                with status_lock:
                    node_status[node_name].update({ "cpu": round(cpu_percent, 2), "memory": round(memory_usage, 2) })
            else:
                mark_node_down(node_name, f"Status: {container.status}")

        except docker.errors.NotFound:
            mark_node_down(node_name, "Not Found")
        except Exception as e:
            mark_node_down(node_name, f"Error: {e}")

@app.route('/get_fog_node', methods=['GET'])
def get_fog_node():
    update_all_node_statuses()
    with status_lock:
        active_nodes = {node: info for node, info in node_status.items() if info["status"] == "UP"}
    if not active_nodes:
        return jsonify({"error": "No active fog nodes available"}), 503
    
    # --- Simple Reactive Logic ---
    chosen_node = min(active_nodes, key=lambda node: active_nodes[node]['cpu'])
    
    print(f"STATUS_UPDATE::{node_status}")
    return jsonify({"fog_node_host": chosen_node})

@app.route('/report_failure', methods=['POST'])
def report_failure():
    failed_node = request.get_json().get('node')
    if failed_node:
        mark_node_down(failed_node, "Reported by IoT device")
    return jsonify({"status": "acknowledged"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
