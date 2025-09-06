
import socket
import json
import os
import time

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 9999
NODE_ID = os.uname().nodename
# --- NEW: Control the duration of the CPU-intensive task ---
# This value (in seconds) creates a sustained load instead of a short spike.
# It makes the CPU usage patterns much more stable and realistic for time-series analysis.
WORKLOAD_DURATION_SECONDS = 1.5 

def cpu_intensive_task(duration_seconds):
    """
    Simulates a sustained CPU-heavy workload for a given duration.
    Instead of a recursive function that ends quickly, this loop keeps the CPU busy.
    This is crucial for generating smooth, analyzable time-series data for the LSTM model.
    """
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        # This is a simple mathematical operation to consume CPU cycles.
        _ = 12345 * 54321

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"[{NODE_ID}] Fog node is running and listening on {HOST}:{PORT}")

while True:
    conn, addr = server_socket.accept()
    with conn:
        # Capture the time the request was received
        reception_time = time.time()
        print(f"[{NODE_ID}] Connected by {addr}")
        data = conn.recv(1024)
        
        if data:
            try:
                message = json.loads(data.decode('utf-8'))
                print(f"[{NODE_ID}] Received data: {message}")

                # --- Execute the new, sustained heavy task ---
                print(f"[{NODE_ID}] Starting CPU-intensive task for {WORKLOAD_DURATION_SECONDS} seconds...")
                start_cpu_task = time.time()
                
                cpu_intensive_task(WORKLOAD_DURATION_SECONDS) 
                
                end_cpu_task = time.time()
                print(f"[{NODE_ID}] Finished CPU task in {end_cpu_task - start_cpu_task:.2f} seconds.")

                # Calculate and print the end-to-end latency
                start_time = message.get("timestamp", 0)
                if start_time > 0:
                    # Latency is calculated from the IoT device's timestamp to the moment the request was received.
                    latency = (reception_time - start_time) * 1000
                    print(f"[{NODE_ID}] >> Calculated E2E Latency: {latency:.2f} ms")

            except json.JSONDecodeError:
                print(f"[{NODE_ID}] Received non-JSON data: {data.decode('utf-8')}")
