import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Aggregate and correlate surgery metrics')
parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
parser.add_argument('--vis', action='store_true', help='Enable visualization (default: False)')
parser.add_argument('--separate-plots', action='store_true', help='Save each metric as separate image (default: False)')
args = parser.parse_args()

# Paths
base_dir = args.base_dir
scores_path = os.path.join(base_dir, 'Analysis', 'scores.xlsx')
completion_path = os.path.join(base_dir, 'Analysis', 'completion_times.xlsx')
movements_path = os.path.join(base_dir, 'Analysis', 'body_movements.xlsx')
dist_path = os.path.join(base_dir, 'Analysis', 'hand_dists_allgroups_smoothed_0.0.xlsx')
stability_path = os.path.join(base_dir, 'Analysis', 'hand_stability_scores_procrustes_smoothed.xlsx')
smoothness_path = os.path.join(base_dir, 'Analysis', 'hand_smoothness_scores_sparc.xlsx')
output_path = os.path.join(base_dir, 'Analysis', 'aggregated_metrics.xlsx')

# Groups and order
groups = ['Students', 'Residents', 'Attendings']

# 1. Read scores.xlsx
scores_df = pd.read_excel(scores_path)

# 2. Read completion_times.xlsx (all sheets)
completion_df = pd.read_excel(completion_path, sheet_name=None, header=None, index_col=0)
completion_records = []
for group, df_group in completion_df.items():
    for subj_name in df_group.index:
        completion_records.append({
            'Subject_name': subj_name,
            'Group': group,
            'CompletionTime': df_group.loc[subj_name].values[0]
        })
completion_times_df = pd.DataFrame(completion_records)

# 3. Read body_movements.xlsx (only {Group}_Neck sheets)
movements_df = pd.read_excel(movements_path, sheet_name=None, header=None, index_col=0)
movements_records = []
for group in groups:
    sheet_name = f'{group}_Neck'
    if sheet_name in movements_df:
        df_group = movements_df[sheet_name]
        for subj_name in df_group.index:
            movements_records.append({
                'Subject_name': subj_name,
                'Group': group,
                'BodyMovement': df_group.loc[subj_name].values[0]
            })
movements_data = pd.DataFrame(movements_records)

# 4. Read hand_dists_allgroups_filtered.xlsx (all group sheets)
df_all = pd.read_excel(dist_path, sheet_name=groups, header=0, index_col=0)
dist_records = []
for group in groups:
    df_group = df_all[group]
    for subj_name in df_group.index:
        left_dist = df_group.loc[subj_name].values[0]
        right_dist = df_group.loc[subj_name].values[1]
        dist_records.append({
            'Subject_name': subj_name,
            'Group': group,
            'NonDominant_Distance': left_dist,
            'Dominant_Distance': right_dist
        })
dist_data = pd.DataFrame(dist_records)

# 4. Read hand_stability_scores_procrustes_smoothed.xlsx (all group sheets)
df_all = pd.read_excel(stability_path, sheet_name=groups, header=None, index_col=0)
stability_records = []
for group in groups:
    df_group = df_all[group]
    for subj_name in df_group.index:
        stability_records.append({
            'Subject_name': subj_name,
            'Group': group,
            'NonDominant_Hand_Stability': df_group.loc[subj_name].values[0],
            'Dominant_Hand_Stability': df_group.loc[subj_name].values[1]
        })
stability_data = pd.DataFrame(stability_records)

# 4. Read hand_smoothness_scores_smoothed.xlsx (all group sheets)
df_all = pd.read_excel(smoothness_path, sheet_name=groups, header=0, index_col=0)
smoothness_records = []
for group in groups:
    df_group = df_all[group]
    for subj_name in df_group.index:
        smoothness_records.append({
            'Subject_name': subj_name,
            'Group': group,
            'NonDominant_Hand_Smoothness': df_group.loc[subj_name].values[0],
            'Dominant_Hand_Smoothness': df_group.loc[subj_name].values[1]
        })
smoothness_data = pd.DataFrame(smoothness_records)

# 5. Merge all dataframes on Subject_name and Group
merged = pd.merge(scores_df, completion_times_df, on=['Subject_name', 'Group'], how='inner')
merged = pd.merge(merged, movements_data, on=['Subject_name', 'Group'], how='inner')
merged = pd.merge(merged, dist_data, on=['Subject_name', 'Group'], how='inner')
merged = pd.merge(merged, stability_data, on=['Subject_name', 'Group'], how='inner')
merged = pd.merge(merged, smoothness_data, on=['Subject_name', 'Group'], how='inner')

# 6. Reorder and rename columns
final_df = merged[['Index', 'Group', 'Subject_name', 'Scores_Avg', 'CompletionTime', 'BodyMovement', 'NonDominant_Distance', 'Dominant_Distance',
                   "NonDominant_Hand_Stability", "Dominant_Hand_Stability", "NonDominant_Hand_Smoothness", "Dominant_Hand_Smoothness"]]

# 7. Save to Excel
final_df.to_excel(output_path, index=False)
print(f"Aggregated data saved to {output_path}")

# 7.1. Compute group statistics (mean and std) for each metric
metrics = ['Scores_Avg', 'CompletionTime', 'BodyMovement', 'NonDominant_Distance', 'Dominant_Distance',
           'NonDominant_Hand_Stability', 'Dominant_Hand_Stability', 'NonDominant_Hand_Smoothness', 'Dominant_Hand_Smoothness']
group_stats = []

for group in groups:
    group_data = final_df[final_df['Group'] == group]
    stats_row = {'Group': group}
    
    for metric in metrics:
        mean_val = group_data[metric].mean()
        std_val = group_data[metric].std()
        stats_row[f'{metric}_Mean'] = mean_val
        stats_row[f'{metric}_Std'] = std_val
    
    group_stats.append(stats_row)

group_stats_df = pd.DataFrame(group_stats)

# 7.2. Save both dataframes to the same Excel file with different sheets
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name='Aggregated_Data', index=False)
    group_stats_df.to_excel(writer, sheet_name='Group_Statistics', index=False)

print(f"Aggregated data and group statistics saved to {output_path}")
print("\nGroup Statistics:")
print(group_stats_df.to_string(index=False))

# 8. Compute and print correlation between 'Score' and each of the other columns
corrs = {}
for col in ['CompletionTime', 'BodyMovement', 'NonDominant_Distance', 'Dominant_Distance', 
            'NonDominant_Hand_Stability', 'Dominant_Hand_Stability', 'NonDominant_Hand_Smoothness', 'Dominant_Hand_Smoothness']:
    corr = final_df['Scores_Avg'].corr(final_df[col])
    corrs[col] = corr
    print(f"Correlation between Scores and {col}: {corr:.4f}") 

# 9. Create visualizations (optional)
if args.vis:
    print("\nGenerating visualizations...")
    plot_dir = os.path.join(base_dir, 'Analysis', 'plots_allgroups')
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Extract mean and std values for plotting
    mean_data = {}
    std_data = {}
    for metric in metrics:
        mean_data[metric] = [group_stats_df[group_stats_df['Group'] == group][f'{metric}_Mean'].values[0] 
                            for group in groups]
        std_data[metric] = [group_stats_df[group_stats_df['Group'] == group][f'{metric}_Std'].values[0] 
                           for group in groups]
    
    if args.separate_plots:
        plot_dir = os.path.join(base_dir, 'Analysis', 'plots_allgroups', 'barplots')
        os.makedirs(plot_dir, exist_ok=True)
        # Create separate plots for each metric
        print("Creating separate plots for each metric...")
        
        for metric in metrics:
            # Bar plot with error bars
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            x_pos = np.arange(len(groups))
            bars = ax.bar(x_pos, mean_data[metric], yerr=std_data[metric], 
                         alpha=0.8, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            ax.set_title(f'{metric} by Group', fontweight='bold', fontsize=14)
            ax.set_xlabel('Group', fontsize=12)
            ax.set_ylabel('Average Value ± Std', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value, std_val in zip(bars, mean_data[metric], std_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + height*0.01,
                        f'{value:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save individual plot with error bars
            plot_path = os.path.join(plot_dir, f'{metric}_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  {metric} plot saved to {plot_path}")
    
    else:
        # Create combined plots (original behavior)
        print("Creating combined plots...")
        
        # Combined plots with error bars
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle('Average Metrics by Group', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Plot each metric with error bars
        for i, metric in enumerate(metrics):
            ax = axes[i]
            x_pos = np.arange(len(groups))
            bars = ax.bar(x_pos, mean_data[metric], yerr=std_data[metric], 
                          alpha=0.8, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.set_xlabel('Group')
            ax.set_ylabel('Average Value ± Std')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value, std_val in zip(bars, mean_data[metric], std_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + height*0.01,
                        f'{value:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Remove the empty subplot
        fig.delaxes(axes[9])
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save combined plot with error bars
        plot_path = os.path.join(plot_dir, 'group_metrics_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Combined bar plots saved to {plot_path}")

else:
    print("\nVisualization skipped. Use --vis flag to enable plotting.") 