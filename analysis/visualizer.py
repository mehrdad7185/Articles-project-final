# visualizer.py (Final Unified Version)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def plot_single_scenario(data, scenario_name):
    """
    Creates a detailed, multi-metric plot for a single scenario,
    based on the user's preferred high-quality style.
    """
    print(f"Generating detailed plots for '{scenario_name}' scenario...")
    
    # --- Data Extraction ---
    cpu_data = data[data['metric'] == 'cpu'].copy()
    mem_data = data[data['metric'] == 'memory'].copy()
    latency_data = data[data['metric'] == 'latency'].copy()
    risk_data = data[data['metric'] == 'risk'].copy()

    # --- Plot 1: CPU Time-Series ---
    if not cpu_data.empty:
        # Smooth the CPU data for better readability
        cpu_data['cpu_smooth'] = cpu_data.groupby('node')['value'].transform(lambda s: s.rolling(5, min_periods=1).mean())
        plt.figure(figsize=(20, 8))
        ax = sns.lineplot(x='timestamp', y='cpu_smooth', data=cpu_data, hue='node', style='node', linewidth=2.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_title(f'CPU Load Distribution in {scenario_name} Scenario', fontsize=18)
        ax.set_ylabel('Smoothed CPU Usage (%)')
        ax.set_xlabel('Time (Minute:Second)')
        ax.legend(title='Fog Node')
        # Save the plot with a descriptive filename
        plt.savefig(f'plot_cpu_{scenario_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: plot_cpu_{scenario_name.lower()}.png")
        plt.close()

    # --- Plot 2: Memory Time-Series (if available) ---
    if not mem_data.empty:
        plt.figure(figsize=(20, 8))
        ax = sns.lineplot(x='timestamp', y='value', data=mem_data, hue='node', style='node', linewidth=2.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_title(f'Memory Usage in {scenario_name} Scenario', fontsize=18)
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_xlabel('Time (Minute:Second)')
        ax.legend(title='Fog Node')
        plt.savefig(f'plot_memory_{scenario_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: plot_memory_{scenario_name.lower()}.png")
        plt.close()

    # --- Plot 3: Risk Score Time-Series (if available) ---
    if not risk_data.empty:
        plt.figure(figsize=(20, 8))
        ax = sns.lineplot(x='timestamp', y='value', data=risk_data, hue='node', style='node', linewidth=2.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_title(f'Predicted Failure Risk in {scenario_name} Scenario', fontsize=18)
        ax.set_ylabel('Failure Risk Score (0.0 to 1.0)')
        ax.set_xlabel('Time (Minute:Second)')
        ax.axhline(0.7, ls='--', color='red', label='Risk Threshold (0.7)')
        ax.legend(title='Fog Node')
        plt.savefig(f'plot_risk_{scenario_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: plot_risk_{scenario_name.lower()}.png")
        plt.close()

def plot_comparison(data):
    """
    Creates high-level comparison plots between scenarios.
    """
    print("Generating comparison plots...")
    cpu_data = data[data['metric'] == 'cpu'].copy()
    if not cpu_data.empty:
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="scenario", y="value", hue="node", data=cpu_data, order=['Reactive', 'Proactive'])
        ax.set_title('Comparison of Overall CPU Load Distribution', fontsize=18)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('CPU Usage (%)')
        ax.legend(title='Fog Node')
        plt.savefig('plot_comparison_cpu_boxplot.png', dpi=300, bbox_inches='tight')
        print("Saved: plot_comparison_cpu_boxplot.png")
        plt.close()

if __name__ == '__main__':
    try:
        # Attempt to read the results file
        data = pd.read_csv('results.csv', parse_dates=['timestamp'])
    except FileNotFoundError:
        print("Error: 'results.csv' not found. Please run parser.py first.")
    else:
        # Set the global plot theme
        sns.set_theme(style="darkgrid", palette="colorblind")
        
        # Create detailed plots for each scenario found in the data
        for scenario in data['scenario'].unique():
            scenario_data = data[data['scenario'] == scenario]
            plot_single_scenario(scenario_data, scenario)
            
        # Create comparison plots if more than one scenario exists
        if data['scenario'].nunique() > 1:
            plot_comparison(data)

