# parser.py (Final Unified Version)

import re
import pandas as pd
import os
import ast

def parse_log_file(log_file_path, scenario_name):
    """
    Parses a log file for all available metrics (cpu, memory, latency, risk)
    and assigns a clear scenario name.
    """
    log_line_regex = re.compile(r"^(.*?)\s*\|\s*(.*)")
    latency_regex = re.compile(r".*?>> Calculated E2E Latency: ([\d\.]+)\s*ms")
    records = []
    time_step = 0
    
    print(f"Parsing '{log_file_path}' for scenario '{scenario_name}'...")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_line_regex.match(line)
            if not match:
                continue
            
            container_name = match.group(1).strip()
            message = match.group(2).strip()
            time_step += 1
            current_timestamp = pd.to_datetime('2025-01-01') + pd.to_timedelta(time_step, unit='s')

            # --- 1. Extract Latency from Fog Node logs ---
            if 'fog-node' in container_name:
                latency_match = latency_regex.search(message)
                if latency_match:
                    records.append({
                        "scenario": scenario_name, "metric": "latency", "node": container_name,
                        "value": float(latency_match.group(1)), "timestamp": current_timestamp
                    })

            # --- 2. Extract Metrics from Scheduler logs ---
            if message.startswith("STATUS_UPDATE::"):
                try:
                    data_str = message.replace("STATUS_UPDATE::", "")
                    status_data = ast.literal_eval(data_str)
                    
                    if isinstance(status_data, list): # Proactive format
                        for node_stats in status_data:
                            node_name = node_stats.get('name')
                            if node_name:
                                records.append({"scenario": scenario_name, "metric": "cpu", "node": node_name, "value": node_stats.get('cpu', 0.0), "timestamp": current_timestamp})
                                records.append({"scenario": scenario_name, "metric": "risk", "node": node_name, "value": node_stats.get('risk', 0.0), "timestamp": current_timestamp})
                    
                    elif isinstance(status_data, dict): # Reactive format
                        for node_name, stats in status_data.items():
                             if 'fog-node' in node_name:
                                records.append({"scenario": scenario_name, "metric": "cpu", "node": node_name, "value": stats.get('cpu', 0.0), "timestamp": current_timestamp})
                                records.append({"scenario": scenario_name, "metric": "memory", "node": node_name, "value": stats.get('memory', 0.0), "timestamp": current_timestamp})
                except (ValueError, SyntaxError):
                    pass
    return records

if __name__ == '__main__':
    log_files_to_parse = {
        '../reactive_log.txt': 'Reactive',
        '../proactive_log.txt': 'Proactive',
    }
    
    all_records = []
    for file_path, scenario_name in log_files_to_parse.items():
        if os.path.exists(file_path):
            all_records.extend(parse_log_file(file_path, scenario_name))
        else:
            print(f"Warning: Log file not found at '{file_path}'. Skipping.")
            
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv('results.csv', index=False, mode='w')
        print(f"\nParsing complete. {len(df)} records saved to 'results.csv'")
        print("Sample of the parsed data:")
        print(df.head())

