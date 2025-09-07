
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def create_visualizations(csv_path):
    """
    Reads the final, clean data and creates publication-quality plots.
    This version includes fixes for the NameError and TypeError.
    """
    try:
        data = pd.read_csv(csv_path, parse_dates=['timestamp'])
    except (FileNotFoundError, KeyError):
        print(f"Error reading '{csv_path}'. Please run the parser.py script first.")
        return

    sns.set_theme(style="darkgrid", palette="colorblind")

    # --- Data Extraction ---
    latency_data = data[data['metric'] == 'latency']
    failure_scenario_data = data[data['scenario'] == 'Resource-Aware (Failure)']
    cpu_data = failure_scenario_data[failure_scenario_data['metric'] == 'cpu']
    mem_data = failure_scenario_data[failure_scenario_data['metric'] == 'memory'].copy() # Use .copy() to avoid warnings
    event_data = failure_scenario_data[failure_scenario_data['metric'] == 'event']

    # --- Plot 1: Latency Distribution ---
    if not latency_data.empty:
        
        latency_data = latency_data.copy()
        latency_data['value'] = pd.to_numeric(latency_data['value'], errors='coerce')

        plt.figure(figsize=(10, 8)) # Increased figure height for better readability

        # --- KEY CHANGE: Replace sns.boxplot with sns.violinplot ---
        # A violin plot is much better at showing the distribution shape when there are outliers.
        ax = sns.violinplot(x='scenario', y='value', data=latency_data, inner="quartile")
        
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
        ax.set_title('Service Response Time Distribution per Scenario', fontsize=16)
        ax.set_ylabel('Response Time (ms)')
        ax.set_xlabel('Scenario')
        # Add a grid for better value reading
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.savefig('plot_latency_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: plot_latency_distribution.png")

    # --- Plot 2: CPU Usage Time-Series with Rolling Average ---
    if not cpu_data.empty:
        plt.figure(figsize=(20, 8))
        cpu_data = cpu_data.copy()
        cpu_data['cpu_smooth'] = cpu_data.groupby('node')['value'].transform(lambda s: s.rolling(15, min_periods=1).mean())
        ax = sns.lineplot(x='timestamp', y='cpu_smooth', data=cpu_data, hue='node', style='node', linewidth=2.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_title('Smoothed CPU Usage During Failure & Recovery Scenario', fontsize=18)
        ax.set_ylabel('CPU Usage (%) (Rolling Average)')
        # --- KEY FIX: Corrected the typo ---
        ax.set_xlabel('Time (Minute:Second)')
        ax.legend(title='Fog Node')
        plt.savefig('plot_cpu_timeseries.png', dpi=300, bbox_inches='tight')
        print("Saved: plot_cpu_timeseries.png")

    # --- Plot 3: Memory Usage with Clean Annotations ---
    if not mem_data.empty:
        # --- KEY FIX: Ensure the 'value' column is numeric to prevent TypeError ---
        mem_data['value'] = pd.to_numeric(mem_data['value'], errors='coerce')
        mem_data.dropna(subset=['value'], inplace=True) # Drop rows where conversion failed

        plt.figure(figsize=(20, 8))
        ax = sns.lineplot(x='timestamp', y='value', data=mem_data, hue='node', style='node', markers=True, linewidth=2)
        
        # Manually set the y-axis to be more intuitive
        y_max_val = mem_data['value'].max()
        ax.set_ylim(bottom=-0.5, top=y_max_val * 1.1) 

        y_min, y_max = ax.get_ylim()

        if not event_data.empty:
            for _, event in event_data.iterrows():
                node_short_name = event['node'].replace('fog-node-', 'FN-')
                
                if event['value'] == 'DOWN':
                    y_offset_text = y_max * 0.01 
                    y_offset_arrow = y_max * 0.1
                    color = 'red'
                else: # 'UP'
                    y_offset_text = y_max * 0.08
                    y_offset_arrow = y_max * 0.1
                    color = 'green'

                ax.annotate(f"{node_short_name} {event['value']}",
                            xy=(event['timestamp'], y_offset_arrow),
                            xytext=(event['timestamp'], y_offset_text),
                            arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=5),
                            ha='center', va='bottom', fontsize=11, color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_title('Memory Usage During Failure & Recovery Scenario', fontsize=18)
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_xlabel('Time (Minute:Second)')
        ax.legend(title='Fog Node')
        plt.savefig('plot_memory_final.png', dpi=300, bbox_inches='tight')
        print("Saved: plot_memory_final.png")

    plt.show()

if __name__ == '__main__':
    create_visualizations('results.csv')