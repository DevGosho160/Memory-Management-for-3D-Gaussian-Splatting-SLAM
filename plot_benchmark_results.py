import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(csv_file="tum_benchmark_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    
    # Ensure VRAM Limit is treated as categorical/string so 'None' is handled gracefully
    df['VRAM Limit (GB)'] = df['VRAM Limit (GB)'].fillna('None').astype(str).str.replace('.0', '', regex=False)
    df['VRAM Limit (GB)'] = df['VRAM Limit (GB)'].replace('nan', 'None')
    
    # Set the order for VRAM limits for consistent plotting
    unique_vram = [x for x in df['VRAM Limit (GB)'].unique() if x != 'None']
    unique_vram.sort(key=float)
    vram_order = unique_vram + (['None'] if 'None' in df['VRAM Limit (GB)'].values else [])

    # Set up seaborn style
    sns.set_theme(style="whitegrid")
    
    output_files = []

    # 1. Pruning Method vs ATE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Pruning Method", y="ATE", hue="VRAM Limit (GB)", hue_order=vram_order, palette="viridis", errorbar=None)
    plt.title("Pruning Method vs ATE by VRAM Limit")
    plt.ylabel("ATE (meters)")
    plt.legend(title="VRAM Limit (GB)")
    out1 = "plot_pruning_vs_ate.png"
    plt.savefig(out1, bbox_inches='tight')
    plt.close()
    output_files.append(out1)

    # 2. Pruning Method vs Processing time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Pruning Method", y="Processing Time (s)", hue="VRAM Limit (GB)", hue_order=vram_order, palette="viridis", errorbar=None)
    plt.title("Pruning Method vs Processing Time by VRAM Limit")
    plt.ylabel("Processing Time (s)")
    plt.legend(title="VRAM Limit (GB)")
    out2 = "plot_pruning_vs_time.png"
    plt.savefig(out2, bbox_inches='tight')
    plt.close()
    output_files.append(out2)

    # 3. Memory limit vs ATE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="VRAM Limit (GB)", y="ATE", hue="Pruning Method", order=vram_order, errorbar=None)
    plt.title("Memory Limit vs ATE by Pruning Method")
    plt.ylabel("ATE (meters)")
    plt.legend(title="Pruning Method")
    out3 = "plot_memory_vs_ate.png"
    plt.savefig(out3, bbox_inches='tight')
    plt.close()
    output_files.append(out3)

    # 4. Memory Limit vs Processing time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="VRAM Limit (GB)", y="Processing Time (s)", hue="Pruning Method", order=vram_order, errorbar=None)
    plt.title("Memory Limit vs Processing Time by Pruning Method")
    plt.ylabel("Processing Time (s)")
    plt.legend(title="Pruning Method")
    out4 = "plot_memory_vs_time.png"
    plt.savefig(out4, bbox_inches='tight')
    plt.close()
    output_files.append(out4)

    print("Successfully generated graphs:")
    for f in output_files:
        print(f"  - {f}")

if __name__ == "__main__":
    generate_plots()
