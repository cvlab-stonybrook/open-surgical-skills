import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_metric_sheets(file_path, groups, subject_col='Subject', subject_col_rename='Subject_name'):
    """
    Reads an Excel file with one sheet per group and concatenates them into a single DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return pd.DataFrame()
    
    df_all = pd.read_excel(file_path, sheet_name=groups, header=0)
    records = []
    for group, df_group in df_all.items():
        # Rename subject column for consistent merging
        if subject_col in df_group.columns:
            df_group = df_group.rename(columns={subject_col: subject_col_rename})
        
        df_group['Group'] = group
        records.append(df_group)
    
    if not records:
        return pd.DataFrame()
        
    return pd.concat(records, ignore_index=True)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Aggregate and correlate egocentric surgery metrics')
    parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Data and Analysis folders')
    parser.add_argument('--vis', action='store_true', help='Enable visualization (default: False)')
    parser.add_argument('--separate-plots', action='store_true', help='Save each metric as separate image (default: False)')
    args = parser.parse_args()

    # Paths
    base_dir = args.base_dir
    analysis_dir = os.path.join(base_dir, 'Analysis')
    
    # Input files
    scores_path = os.path.join(analysis_dir, 'scores.xlsx')
    completion_path = os.path.join(analysis_dir, 'completion_times.xlsx')
    head_move_path = os.path.join(analysis_dir, 'head_movement_avg_per_cycle.xlsx')
    #grip_stability_path = os.path.join(analysis_dir, 'hand_grip_stability_scores_ego.xlsx')
    smoothness_path = os.path.join(analysis_dir, 'hand_smoothness_scores_sparc_ego.xlsx')
    stability_path = os.path.join(analysis_dir, 'hand_stability_scores_procrustes_ego.xlsx')
    
    # Output file
    output_path = os.path.join(analysis_dir, 'aggregated_metrics_ego.xlsx')

    # Groups and order
    groups = ['Students', 'Residents', 'Attendings']

    # 1. Read scores.xlsx
    if not os.path.exists(scores_path):
        print(f"ERROR: Base scores file not found at {scores_path}. Cannot proceed.")
        return
    scores_df = pd.read_excel(scores_path)

    # 2. Read completion_times.xlsx
    if not os.path.exists(completion_path):
        print(f"Warning: Completion times file not found at {completion_path}. Skipping.")
        completion_times_df = pd.DataFrame(columns=['Subject_name', 'Group', 'CompletionTime'])
    else:
        completion_df_sheets = pd.read_excel(completion_path, sheet_name=None, header=None, index_col=0)
        completion_records = []
        for group, df_group in completion_df_sheets.items():
            for subj_name, row in df_group.iterrows():
                completion_records.append({'Subject_name': subj_name, 'Group': group, 'CompletionTime': row.values[0]})
        completion_times_df = pd.DataFrame(completion_records)

    # 3. Read all egocentric metric files
    head_move_data = read_metric_sheets(head_move_path, groups)
    #grip_stability_data = read_metric_sheets(grip_stability_path, groups)
    smoothness_data = read_metric_sheets(smoothness_path, groups)
    stability_data = read_metric_sheets(stability_path, groups)

    # 4. Merge all dataframes on Subject_name and Group
    merged = pd.merge(scores_df, completion_times_df, on=['Subject_name', 'Group'], how='left')
    
    # Define a list of dataframes to merge
    metric_dfs = [head_move_data, smoothness_data, stability_data]
    for df in metric_dfs:
        if not df.empty:
            merged = pd.merge(merged, df, on=['Subject_name', 'Group'], how='left')

    # 5. Reorder and select final columns
    final_cols = [
        'Index', 'Group', 'Subject_name', 'Scores_Avg', 'CompletionTime',
        'Avg_Large_Head_Movements_Per_Cycle',
        'Dominant_Hand_Smoothness', 'NonDominant_Hand_Smoothness',
        'Dominant_Hand_Stability', 'NonDominant_Hand_Stability'
    ]
    # Ensure only existing columns are selected
    final_cols_exist = [col for col in final_cols if col in merged.columns]
    final_df = merged[final_cols_exist]

    # 6. Compute group statistics (mean and std)
    metrics_to_analyze = [col for col in final_cols if col not in ['Index', 'Group', 'Subject_name']]
    group_stats = []

    for group in groups:
        group_data = final_df[final_df['Group'] == group]
        stats_row = {'Group': group}
        for metric in metrics_to_analyze:
            stats_row[f'{metric}_Mean'] = group_data[metric].mean()
            stats_row[f'{metric}_Std'] = group_data[metric].std()
        group_stats.append(stats_row)
    group_stats_df = pd.DataFrame(group_stats)

    # 7. Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Aggregated_Data_Ego', index=False)
        group_stats_df.to_excel(writer, sheet_name='Group_Statistics_Ego', index=False)
    print(f"Aggregated egocentric data and stats saved to {output_path}")
    print("\nGroup Statistics (Egocentric):")
    print(group_stats_df.to_string(index=False))

    # 8. Compute and print correlations
    print("\nCorrelations with Scores_Avg (Egocentric):")
    for col in metrics_to_analyze:
        if col != 'Scores_Avg':
            corr = final_df['Scores_Avg'].corr(final_df[col])
            print(f"  - {col}: {corr:.4f}")

    # 9. Create visualizations
    if args.vis:
        print("\nGenerating visualizations...")
        plot_dir = os.path.join(analysis_dir, 'plots_ego')
        os.makedirs(plot_dir, exist_ok=True)
        plt.style.use('default')

        # Extract mean and std values for plotting
        mean_data = {m: group_stats_df[f'{m}_Mean'].values for m in metrics_to_analyze}
        std_data = {m: group_stats_df[f'{m}_Std'].values for m in metrics_to_analyze}
        
        num_metrics = len(metrics_to_analyze)
        if args.separate_plots:
            # Create separate plots for each metric
            for metric in metrics_to_analyze:
                fig, ax = plt.subplots(figsize=(8, 6))
                x_pos = np.arange(len(groups))
                bars = ax.bar(x_pos, mean_data[metric], yerr=std_data[metric], alpha=0.8, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_title(f'{metric} by Group', fontweight='bold')
                ax.set_ylabel('Average Value ± Std')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups)
                plt.tight_layout()
                plot_path = os.path.join(plot_dir, f'{metric}_comparison_ego.png')
                plt.savefig(plot_path, dpi=300)
                plt.close(fig)
                print(f"  Plot saved to {plot_path}")
        else:
            # Create combined plot
            nrows = 3
            ncols = 4
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))
            fig.suptitle('Average Egocentric Metrics by Group', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            for i, metric in enumerate(metrics_to_analyze):
                ax = axes[i]
                x_pos = np.arange(len(groups))
                ax.bar(x_pos, mean_data[metric], yerr=std_data[metric], alpha=0.8, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_title(metric, fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups, rotation=45, ha="right")
            
            # Hide unused subplots
            for i in range(num_metrics, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(plot_dir, 'group_metrics_comparison_ego.png')
            plt.savefig(plot_path, dpi=300)
            print(f"  Combined plot saved to {plot_path}")
    else:
        print("\nVisualization skipped. Use --vis flag to enable plotting.")

if __name__ == '__main__':
    main()
