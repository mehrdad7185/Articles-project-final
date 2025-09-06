
import socket
import time
import json
import random
import requests
import os # Import the 'os' module

# --- Configuration ---
SCHEDULER_URL = "http://scheduler:5000"
FOG_PORT = 9999
# --- NEW: Get a unique ID from the container's hostname ---
# This makes logs much easier to read when using replicas.
DEVICE_ID = os.environ.get('HOSTNAME', 'iot-device-unknown')

def get_target_fog_node():
    """Gets the target fog node from the scheduler."""
    try:
        response = requests.get(f"{SCHEDULER_URL}/get_fog_node")
        response.raise_for_status()
        data = response.json()
        return data.get("fog_node_host")
    except requests.exceptions.RequestException as e:
        print(f"[{DEVICE_ID}] Could not connect to scheduler: {e}")
        return None

def report_failure_to_scheduler(failed_node):
    """Reports a connection failure to the scheduler."""
    try:
        requests.post(f"{SCHEDULER_URL}/report_failure", json={"node": failed_node})
        print(f"[{DEVICE_ID}] Successfully reported failure of node '{failed_node}' to scheduler.")
    except requests.exceptions.RequestException as e:
        print(f"[{DEVICE_ID}] Could not report failure to scheduler: {e}")

while True:
    target_host = get_target_fog_node()
    if not target_host:
        print(f"[{DEVICE_ID}] Could not get a fog node from scheduler. Retrying...")
        time.sleep(5)
        continue

    print(f"[{DEVICE_ID}] Scheduler assigned: '{target_host}'. Sending data...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((target_host, FOG_PORT))
            
            start_time = time.time()
            payload = {
                "device_id": DEVICE_ID,
                "timestamp": start_time
            }
            message = json.dumps(payload).encode('utf-8')
            s.sendall(message)
            print(f"[{DEVICE_ID}] Successfully sent data to {target_host}")

    except Exception as e:
        print(f"[{DEVICE_ID}] FAILED to connect to {target_host}: {e}")
        report_failure_to_scheduler(target_host)
    
    time.sleep(random.uniform(3, 6)) # Added some randomness to requests