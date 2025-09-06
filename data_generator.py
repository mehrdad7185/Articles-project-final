# data_generator.py

import subprocess
import threading
import time
import re
import pandas as pd
import ast
import os

# --- Configuration ---
SIMULATION_DURATION_SECONDS = 300   # Total time to run the simulation (e.g., 5 minutes)
NODE_TO_FAIL = "fog-node-2"         # The container name of the node we want to fail
FAILURE_TIME_SECONDS = 180          # When the failure should occur (e.g., 3 minutes into the simulation)
PRE_FAILURE_WINDOW_SECONDS = 60     # How many seconds before the failure to label as "imminent failure" (label=1)
LOG_FILE_PATH = "simulation_log_for_dataset.txt" # A temporary file to store logs
OUTPUT_CSV_PATH = "dataset.csv"     # The final output dataset

# --- Regular Expressions for parsing ---
LOG_LINE_REGEX = re.compile(r"^(.*?)\s*\|\s*(.*)")
LATENCY_REGEX = re.compile(r".*?>> Calculated E2E Latency: ([\d\.]+)\s*ms")

def fail_node_task():
    """
    A thread task that waits for a specific time and then stops a container.
    """
    print(f"\n[INFO] Failure simulation is scheduled for '{NODE_TO_FAIL}' at T+{FAILURE_TIME_SECONDS} seconds.\n")
    time.sleep(FAILURE_TIME_SECONDS)
    print(f"\n[INFO] >>> Triggering failure on '{NODE_TO_FAIL}' now! <<<\n")
    try:
        # Use 'docker stop' to simulate a sudden failure
        subprocess.run(["docker", "stop", NODE_TO_FAIL], check=True)
        print(f"\n[INFO] Successfully stopped '{NODE_TO_FAIL}'.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to stop container '{NODE_TO_FAIL}': {e}\n")

def parse_logs_to_dataframe(log_file_path):
    """
    Parses the entire simulation log file and converts it into a structured pandas DataFrame.
    """
    records = []
    print(f"[INFO] Starting to parse the log file: {log_file_path}")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_match = LOG_LINE_REGEX.match(line)
            if not line_match:
                continue
            
            # Use the container name directly from the log without modification
            container_name = line_match.group(1).strip()
            message = line_match.group(2).strip()
            
            # Extract latency if present
            latency_match = LATENCY_REGEX.search(message)
            if latency_match:
                records.append({
                    "timestamp": pd.to_datetime('now', utc=True),
                    "node_id": container_name,
                    "metric": "latency",
                    "value": float(latency_match.group(1))
                })

            # Extract CPU and Memory from scheduler status updates
            if message.startswith("STATUS_UPDATE::"):
                try:
                    # Isolate the dictionary string part of the message
                    dict_str = message.replace("STATUS_UPDATE::", "")
                    status_dict = ast.literal_eval(dict_str)
                    
                    for node_name, stats in status_dict.items():
                        if 'fog-node' in node_name:
                            timestamp = pd.to_datetime('now', utc=True)
                            records.append({
                                "timestamp": timestamp, "node_id": node_name,
                                "metric": "cpu", "value": stats.get('cpu', 0.0)
                            })
                            records.append({
                                "timestamp": timestamp, "node_id": node_name,
                                "metric": "memory", "value": stats.get('memory', 0.0)
                            })
                except (ValueError, SyntaxError) as e:
                    print(f"[WARNING] Could not parse a status update line. Error: {e}")
                    continue

    if not records:
        print("[ERROR] No valid records were parsed. The log file might be empty or in an unexpected format.")
        return pd.DataFrame()

    # Convert the list of records to a DataFrame and pivot it
    df = pd.DataFrame(records)
    df_pivot = df.pivot_table(index=['timestamp', 'node_id'], columns='metric', values='value').reset_index()
    
    df_pivot = df_pivot.sort_values(by=['node_id', 'timestamp'])
    
    # Use modern .ffill() to forward-fill missing values (e.g., latency is reported less often)
    df_pivot[['cpu', 'memory', 'latency']] = df_pivot.groupby('node_id')[['cpu', 'memory', 'latency']].ffill()
    df_pivot = df_pivot.dropna().reset_index(drop=True)
    
    print("[INFO] Log parsing complete. DataFrame created.")
    return df_pivot

def label_data(df):
    """
    Adds the 'is_failure_imminent' column to the DataFrame based on the failure event.
    """
    print("[INFO] Labeling data for failure prediction...")
    df['is_failure_imminent'] = 0
    
    if df.empty: return df
    # Approximate the failure time by taking the last timestamp in the dataset
    end_of_simulation_time = df['timestamp'].max()
    
    # Calculate the time window for imminent failure
    start_of_imminent_window = end_of_simulation_time - pd.to_timedelta(PRE_FAILURE_WINDOW_SECONDS, unit='s')

    # Define the condition for records to be labeled as 'imminent failure'
    imminent_condition = (
        (df['node_id'] == NODE_TO_FAIL) &
        (df['timestamp'] >= start_of_imminent_window)
    )
    
    # Apply the label
    df.loc[imminent_condition, 'is_failure_imminent'] = 1
    
    count = df['is_failure_imminent'].sum()
    print(f"[INFO] Labeling complete. Found {count} data points marked as 'imminent failure'.")
    return df

if __name__ == '__main__':
    # Clean up previous run files
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
        
    # Start the failure-inducing thread in the background
    failure_thread = threading.Thread(target=fail_node_task)
    failure_thread.start()

    print("[INFO] Starting docker-compose simulation...")
    try:
        # Open a file to stream all docker-compose logs into
        with open(LOG_FILE_PATH, "w") as log_file:
            # Use 'docker compose' (with a space) as it's the current standard
            process = subprocess.Popen(
                ["docker", "compose", "up", "--build"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            # Wait for the simulation to run for the specified duration
            process.wait(timeout=SIMULATION_DURATION_SECONDS)

    except subprocess.TimeoutExpired:
        print(f"[INFO] Simulation time of {SIMULATION_DURATION_SECONDS} seconds reached. Terminating...")
        process.terminate()
    except Exception as e:
        print(f"[ERROR] An error occurred during simulation: {e}")
    finally:
        # Ensure all containers are stopped and removed cleanly
        print("[INFO] Cleaning up and stopping all containers...")
        subprocess.run(["docker", "compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    
    # Parse the collected logs
    final_df = parse_logs_to_dataframe(LOG_FILE_PATH)

    # If parsing was successful, label the data and save it
    if not final_df.empty:
        labeled_df = label_data(final_df)
        labeled_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
    # Final check to confirm the file was created and report results
    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"\n[SUCCESS] Dataset successfully created at '{OUTPUT_CSV_PATH}'")
        print("Sample of the final data:")
        print(pd.read_csv(OUTPUT_CSV_PATH).head())
        print("\nDistribution of labels:")
        print(pd.read_csv(OUTPUT_CSV_PATH)['is_failure_imminent'].value_counts())
    else:
        print("\n[FAILURE] The final dataset.csv was NOT created. There might have been no data to parse.")