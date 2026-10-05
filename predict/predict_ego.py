import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, cohen_kappa_score

# --- New imports for PyTorch model ---
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

# --- PyTorch Custom Neural Network with Sigmoid Output ---
# This allows for building a custom NN architecture and wrapping it
# to be compatible with the scikit-learn ecosystem (e.g., for cross-validation).

class Net(nn.Module):
    """A simple feed-forward neural network with a final sigmoid layer."""
    def __init__(self, input_size, hidden_sizes=(10, 5)):
        super(Net, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn wrapper for a PyTorch neural network regressor."""
    def __init__(self, input_size, hidden_sizes=(10, 5), epochs=1000, lr=0.001, random_state=42):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        
        # --- GPU support ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        
    def fit(self, X, y):
        # Build a new network on every call, so that each cross-validation fold is trained from scratch
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.model = Net(input_size=self.input_size, hidden_sizes=self.hidden_sizes).to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy().ravel()


# Path to aggregated data
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, choices=['Linear Regression', 'SVM Regression', 'RandomForest', 'NN_pytorch'])
parser.add_argument('--base_dir', type=str, required=True, help='dataset root containing the Analysis folder with the aggregated metrics')
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()


base_dir = args.base_dir
input_path = os.path.join(base_dir, 'Analysis', 'aggregated_metrics_ego.xlsx')
save_dir = os.path.join(base_dir, 'Analysis', 'Prediction_Ego')
os.makedirs(save_dir, exist_ok=True)
# Load data
df = pd.read_excel(input_path)

# Features and target
features_selected = [
    #'CompletionTime', 
    'Avg_Large_Head_Movements_Per_Cycle',
    'Dominant_Hand_Smoothness', 
    'NonDominant_Hand_Smoothness', 
    'NonDominant_Hand_Stability',
    'Dominant_Hand_Stability'
]
# 'Dominant_Hand_Stability', 
# Filter out features that may not exist in the dataframe
features_selected = [f for f in features_selected if f in df.columns]

print(f"Using features: {features_selected}")
X = df[features_selected].values
y = df['Scores_Avg'].values
subject_info = df[['Group', 'Subject_name']].values

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Models to test
models = {
    'Linear Regression': LinearRegression(),
    'SVM Regression': SVR(kernel='rbf'),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=25),
    'NN_pytorch': PyTorchRegressor(
        input_size=X.shape[1],
        hidden_sizes=(10, 5),
        epochs=2000,
        lr=0.0005,
        random_state=25
    )
}
if args.model is not None:
    models = {args.model: models[args.model]}

# Repeated Stratified K-Fold Cross-Validation for a highly robust evaluation.
n_splits = 4
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
num_folds = rskf.get_n_splits(X_scaled, df['Group'])

print(f"Model evaluation ({n_splits}-fold repeated stratified cross-validation, repeated {n_repeats} times):")
for name, model in models.items():
    print("--------------------------------")
    print("Predicting using model: ", name)
    all_y_test = []
    all_y_pred = []
    
    total_cm = np.zeros((5, 5))

    if name == 'RandomForest':
        feature_importances = []

    for eval_idx, (train_idx, test_idx) in enumerate(rskf.split(X_scaled, df['Group'])):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        y_train_scaled = y_scaled[train_idx].ravel()

        subject_info_train, subject_info_test = subject_info[train_idx], subject_info[test_idx]
        
        model.fit(X_train, y_train_scaled)
        
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        y_pred = np.clip(y_pred, 1, 5)
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        y_pred_rounded = np.round(y_pred).clip(1, 5)
        cm = confusion_matrix(y_test.astype(int), y_pred_rounded.astype(int), labels=np.arange(1, 6))
        total_cm += cm

        if name == 'RandomForest':
            feature_importances.append(model.feature_importances_)

    # --- Calculate overall metrics after loop ---
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_y_pred_rounded = np.round(all_y_pred).clip(1, 5)
    
    # Overall Regression Metrics
    final_r2 = r2_score(all_y_test, all_y_pred)
    final_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
    final_mae = mean_absolute_error(all_y_test, all_y_pred)
    
    # Overall Ordinal/Classification Metrics
    final_srocc, _ = spearmanr(all_y_test, all_y_pred)
    final_acc_within_1 = np.mean(np.abs(all_y_test - all_y_pred_rounded) <= 1)
    final_acc_within_05 = np.mean(np.abs(all_y_test - all_y_pred_rounded) <= 0.5)

    # --- Print Results ---
    print(f"\n{name} - Performance Summary:")
    print(f"  R^2 Score:              {final_r2:.4f}")
    print(f"  MAE:                    {final_mae:.4f}")
    print(f"  Spearman's Rho (SROCC): {final_srocc:.4f}")
    print(f"  Accuracy (within +/-1): {final_acc_within_1:.4f}")
    print(f"  Accuracy (within +/-0.5): {final_acc_within_05:.4f}")
    
    if name == 'RandomForest':
        mean_importances = np.mean(feature_importances, axis=0)
        sorted_indices = np.argsort(mean_importances)[::-1]
        print("  Feature Importances:")
        for i in sorted_indices:
            print(f"    {features_selected[i]}: {mean_importances[i]:.4f}")
    
    
    results = pd.DataFrame({
        'Model': [name],
        'R2': [final_r2],
        'RMSE': [final_rmse],
        'MAE': [final_mae],
        'SROCC': [final_srocc],
        'Accuracy (within +/-1)': [final_acc_within_1],
        'Accuracy (within +/-0.5)': [final_acc_within_05]
    })
    results.to_csv(os.path.join(save_dir, f'{name}_results_ego.csv'), index=False)
    
    if args.vis:
        # --- Generate and Save Plots ---
        # Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.regplot(x=all_y_test, y=all_y_pred, scatter_kws={'alpha':0.6})
        plt.xlabel("True Scores")
        plt.ylabel("Predicted Scores")
        plt.title(f'{name}: True vs. Predicted Scores (Ego)')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{name}_scatter_plot_ego.png'))
        plt.close()
        
        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=np.arange(1,6), yticklabels=np.arange(1,6),
                    annot_kws={"size": 18})
        plt.xlabel("Predicted Level", fontsize=20)
        plt.ylabel("True Level", fontsize=20)
        plt.title(f'Normalized Confusion Matrix', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.png'), dpi=300)
        plt.close() 
