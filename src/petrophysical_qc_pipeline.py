"""
Well Log QC, AI Reconstruction, and Electrofacies Classification
Using the Equinor Volve Dataset (Well 15/9-19)
"""

# ============================================================
# SECTION 0: Install & Import Core Libraries
# ============================================================
# Run this line in your terminal or notebook cell if needed:
# !pip install lasio pandas matplotlib scikit-learn

import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1: Load the LAS File
# ============================================================

# URL to a real well from the Equinor Volve Dataset (Well 15/9-19)
las_url = "https://raw.githubusercontent.com/andymcdgeo/Petrophysics-Python-Series/master/Data/15-9-19_SR_COMP.LAS"

# Load the LAS file using lasio
las = lasio.read(las_url)

# Convert to a Pandas DataFrame
df = las.df()

# The depth is currently the index. Let's make it a column for easier plotting.
df.reset_index(inplace=True)
df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

# Let's look at the columns and the first 5 rows
print("Well Logs Available:", df.columns.tolist())
print(df.head())

# ============================================================
# SECTION 2: Washout Detection & Wireline Log QC
# ============================================================

# 1. Define Bit Size and Calculate Washout
BIT_SIZE = 8.5  # Assuming an 8.5 inch bit for this section
TOLERANCE = 1.0  # 1 inch tolerance

# Create a Delta Caliper column
df['DELTA_CALI'] = df['CALI'] - BIT_SIZE

# Flag washouts: True if Caliper is greater than Bit Size + Tolerance
df['WASHOUT'] = df['CALI'] > (BIT_SIZE + TOLERANCE)

# Filter data to a specific depth range for a clear visualization
df_plot = df[(df['DEPTH'] > 3300) & (df['DEPTH'] < 3600)].copy()

# 2. Plotting the Wireline Logs
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), sharey=True)
fig.suptitle('Automated Wireline Log QC: Washout Detection', fontsize=16, fontweight='bold', y=1.02)

# Track 1: Gamma Ray
ax[0].plot(df_plot['GR'], df_plot['DEPTH'], color='green')
ax[0].set_xlabel("Gamma Ray (API)")
ax[0].set_xlim(0, 150)
ax[0].set_ylabel("Depth (m)")
ax[0].grid(True, linestyle='--', alpha=0.7)

# Track 2: Caliper vs Bit Size
ax[1].plot(df_plot['CALI'], df_plot['DEPTH'], color='black', label='Caliper')
ax[1].axvline(x=BIT_SIZE, color='blue', linestyle='--', label='Bit Size')
ax[1].set_xlabel("Caliper (in)")
ax[1].set_xlim(6, 16)
ax[1].legend(loc='upper right')
ax[1].grid(True, linestyle='--', alpha=0.7)

# Shade the washout on the Caliper track
ax[1].fill_betweenx(df_plot['DEPTH'], BIT_SIZE, df_plot['CALI'],
                    where=(df_plot['CALI'] > BIT_SIZE), color='yellow', alpha=0.5)

# Track 3: Bulk Density (DEN)
ax[2].plot(df_plot['DEN'], df_plot['DEPTH'], color='red')
ax[2].set_xlabel("Bulk Density (g/cc)")
ax[2].set_xlim(1.95, 2.95)
ax[2].invert_xaxis()  # Density is traditionally plotted right-to-left
ax[2].grid(True, linestyle='--', alpha=0.7)

# Shade the bad Density data based on the Washout Flag
for ax_idx in range(3):
    ax[ax_idx].fill_betweenx(df_plot['DEPTH'], 0, 1,
                             where=df_plot['WASHOUT'],
                             color='red', alpha=0.15, transform=ax[ax_idx].get_yaxis_transform())
    ax[ax_idx].invert_yaxis()  # Depth goes down

plt.tight_layout()
plt.show()

# ============================================================
# SECTION 3: AI Log Reconstruction using Random Forest
# ============================================================

from sklearn.ensemble import RandomForestRegressor

# 1. Prepare the Data: Drop rows with missing values in our key columns
df_ml = df.dropna(subset=['GR', 'AC', 'DEN', 'CALI']).copy()

# 2. Split into "Good Data" (Training) and "Bad Data" (To be reconstructed)
df_good = df_ml[df_ml['WASHOUT'] == False]

# Features (Inputs) and Target (Output)
features = ['GR', 'AC']  # We use Sonic (AC) and Gamma Ray to predict Density
X_train = df_good[features]
y_train = df_good['DEN']

# 3. Initialize and Train the Random Forest Machine Learning Model
print("Training Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
print("Training Complete!")

# 4. Predict Density for the ENTIRE well
df_ml['DEN_PREDICTED'] = rf_model.predict(df_ml[features])

# 5. Create a "Final" Reconstructed Log: Use Original DEN if hole is good, use ML DEN if washed out
df_ml['DEN_RECONSTRUCTED'] = np.where(df_ml['WASHOUT'] == True,
                                       df_ml['DEN_PREDICTED'],
                                       df_ml['DEN'])

# Visualization: Before and After
df_plot_ml = df_ml[(df_ml['DEPTH'] > 3300) & (df_ml['DEPTH'] < 3600)].copy()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), sharey=True)
fig.suptitle('AI Log Reconstruction: Random Forest vs Bad Data', fontsize=16, fontweight='bold', y=1.02)

# Track 1: Gamma Ray
ax[0].plot(df_plot_ml['GR'], df_plot_ml['DEPTH'], color='green')
ax[0].set_xlabel("Gamma Ray (API)")
ax[0].set_xlim(0, 150)
ax[0].set_ylabel("Depth (m)")
ax[0].grid(True, linestyle='--', alpha=0.7)

# Track 2: Caliper (Showing the Washout)
ax[1].plot(df_plot_ml['CALI'], df_plot_ml['DEPTH'], color='black', label='Caliper')
ax[1].axvline(x=8.5, color='blue', linestyle='--', label='Bit Size')
ax[1].fill_betweenx(df_plot_ml['DEPTH'], 8.5, df_plot_ml['CALI'],
                    where=(df_plot_ml['CALI'] > 8.5), color='yellow', alpha=0.5)
ax[1].set_xlabel("Caliper (in)")
ax[1].set_xlim(6, 16)
ax[1].grid(True, linestyle='--', alpha=0.7)

# Track 3: Original Bad Density vs Reconstructed ML Density
ax[2].plot(df_plot_ml['DEN'], df_plot_ml['DEPTH'], color='red', alpha=0.4, label='Original (Bad Data)', linestyle='--')
ax[2].plot(df_plot_ml['DEN_RECONSTRUCTED'], df_plot_ml['DEPTH'], color='blue', linewidth=1.5, label='ML Reconstructed')

# Highlight the Washout Zones
ax[2].fill_betweenx(df_plot_ml['DEPTH'], 1.95, 2.95,
                    where=df_plot_ml['WASHOUT'],
                    color='red', alpha=0.1)
ax[2].set_xlabel("Bulk Density (g/cc)")
ax[2].set_xlim(1.95, 2.95)
ax[2].invert_xaxis()
ax[2].legend(loc='upper right')
ax[2].grid(True, linestyle='--', alpha=0.7)

for ax_idx in range(3):
    ax[ax_idx].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================
# SECTION 4: Electrofacies Classification using K-Means
# ============================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 1. Prepare Data for Lithology Classification
features = ['GR', 'DEN', 'NEU', 'AC']

df_facies = df.dropna(subset=features).copy()

# Filter depth for a good visualization window
df_plot = df_facies[(df_facies['DEPTH'] > 3300) & (df_facies['DEPTH'] < 3600)].copy()

# 2. Standardize the Data (Crucial for K-Means distance calculations)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_plot[features])

# 3. Train Unsupervised ML (K-Means) to find 3 distinct Rock Types
print("Running AI Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_plot['FACIES'] = kmeans.fit_predict(scaled_features)

# Sort the facies so the colors align nicely with GR (0=Cleanest, 2=Shaliest)
idx_sort = df_plot.groupby('FACIES')['GR'].mean().sort_values().index
facies_mapping = {old_id: new_id for new_id, old_id in enumerate(idx_sort)}
df_plot['FACIES'] = df_plot['FACIES'].map(facies_mapping)

print("Clustering Complete! Generating Plot...")

# 4. Plotting the Results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), sharey=True)
fig.suptitle('AI-Automated Electrofacies Classification (K-Means)', fontsize=15, fontweight='bold', y=1.02)

# Track 1: Gamma Ray
ax[0].plot(df_plot['GR'], df_plot['DEPTH'], color='green')
ax[0].set_xlabel("Gamma Ray (API)")
ax[0].set_xlim(0, 150)
ax[0].grid(True, linestyle='--', alpha=0.7)

# Track 2: Neutron-Density
ax[1].plot(df_plot['NEU'], df_plot['DEPTH'], color='blue', label='NEU')
ax[1].set_xlabel("Neutron Porosity (v/v)", color='blue')
ax[1].set_xlim(0.6, 0)  # Neutron reversed
ax[1].tick_params(axis='x', colors='blue')

ax1_twin = ax[1].twiny()
ax1_twin.plot(df_plot['DEN'], df_plot['DEPTH'], color='red', label='DEN')
ax1_twin.set_xlabel("Bulk Density (g/cc)", color='red')
ax1_twin.set_xlim(1.95, 2.95)
ax1_twin.invert_xaxis()
ax[1].grid(True, linestyle='--', alpha=0.7)

# Track 3: The AI-Generated Facies Track
cmap_facies = ListedColormap(['gold', 'orange', 'gray'])

im = ax[2].pcolormesh([0, 1], df_plot['DEPTH'], np.vstack((df_plot['FACIES'], df_plot['FACIES'])).T,
                      cmap=cmap_facies, vmin=0, vmax=2)
ax[2].set_xlabel("AI Rock Type")
ax[2].set_xticks([])  # Hide x ticks for a clean look

# Formatting depths
ax[0].invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================
# SECTION 5: Multi-Dimensional Anomaly Detection (LOF + PCA)
# ============================================================

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# 1. Prepare Data for Anomaly Detection
features = ['GR', 'DEN', 'NEU', 'AC']
df_anomaly = df.dropna(subset=features).copy()

# 2. Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_anomaly[features])

# 3. Apply Local Outlier Factor (LOF)
print("Running Local Outlier Factor (LOF) AI...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df_anomaly['OUTLIER_FLAG'] = lof.fit_predict(scaled_data)

# 4. Dimensionality Reduction using PCA
print("Running PCA for Visualization...")
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_data)

df_anomaly['PCA1'] = pca_results[:, 0]
df_anomaly['PCA2'] = pca_results[:, 1]

# 5. Plotting the PCA Scatter Plot
plt.figure(figsize=(10, 7))

plt.scatter(df_anomaly[df_anomaly['OUTLIER_FLAG'] == 1]['PCA1'],
            df_anomaly[df_anomaly['OUTLIER_FLAG'] == 1]['PCA2'],
            c='blue', alpha=0.5, label='Valid Petrophysical Data', edgecolors='w', s=40)

plt.scatter(df_anomaly[df_anomaly['OUTLIER_FLAG'] == -1]['PCA1'],
            df_anomaly[df_anomaly['OUTLIER_FLAG'] == -1]['PCA2'],
            c='red', alpha=0.8, label='Anomalies / Bad Data', edgecolors='k', s=60, marker='X')

plt.title('Multi-Dimensional Anomaly Detection using LOF & PCA', fontsize=15, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("Process Complete! Red 'X' marks highlight non-physical log responses.")

# ============================================================
# SECTION 6: GMM Probabilistic Electrofacies (GR vs Acoustic)
# ============================================================

from sklearn.mixture import GaussianMixture

# 1. Prepare Data for GMM Clustering
features = ['GR', 'AC', 'DEN', 'NEU']

df_gmm = df.dropna(subset=features).copy()

# 2. Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_gmm[features])

# 3. Apply Gaussian Mixture Model (GMM)
print("Running Gaussian Mixture Model (GMM) AI...")
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
df_gmm['GMM_FACIES'] = gmm.fit_predict(scaled_features)

# Sort the facies logically with GR (0=Cleanest, 2=Shaliest)
idx_sort = df_gmm.groupby('GMM_FACIES')['GR'].mean().sort_values().index
facies_mapping = {old_id: new_id for new_id, old_id in enumerate(idx_sort)}
df_gmm['GMM_FACIES'] = df_gmm['GMM_FACIES'].map(facies_mapping)

print("GMM Clustering Complete! Generating Plot...")

# 4. Plotting GR vs AC colored by GMM Facies
plt.figure(figsize=(8, 6))

scatter = plt.scatter(df_gmm['GR'], df_gmm['AC'],
                      c=df_gmm['GMM_FACIES'], cmap='viridis',
                      alpha=0.6, s=15)

plt.title('GMM Probabilistic Electrofacies (GR vs. Acoustic)', fontsize=14, fontweight='bold')
plt.xlabel('Gamma Ray (API)')
plt.ylabel('Acoustic Slowness (us/ft)')
plt.grid(True, linestyle='--', alpha=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('GMM Rock Type (0=Clean, 2=Shale)')
cbar.set_ticks([0, 1, 2])

plt.tight_layout()
plt.show()

print("Process Complete! Data is now grouped into geological facies.")

# ============================================================
# SECTION 7: Facies-Specific AI Reconstruction (GMM + Random Forest)
# ============================================================

# 1. Prepare Data
features = ['GR', 'AC', 'NEU', 'RDEP']
features_gmm = ['GR', 'AC']  # Only robust logs for GMM (unaffected by washouts)
target = 'DEN'

df_clean = df.dropna(subset=features + [target, 'WASHOUT']).copy()

# GMM with GR + AC only (more robust against washouts)
print("1. Running GMM with robust logs (GR + AC)...")
scaler_gmm = StandardScaler()
scaled_features_gmm = scaler_gmm.fit_transform(df_clean[features_gmm])

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, max_iter=200)
df_clean['GMM_FACIES'] = gmm.fit_predict(scaled_features_gmm)

# Sort by GR
idx_sort = df_clean.groupby('GMM_FACIES')['GR'].mean().sort_values().index
facies_mapping = {old_id: new_id for new_id, old_id in enumerate(idx_sort)}
df_clean['GMM_FACIES'] = df_clean['GMM_FACIES'].map(facies_mapping)

print("\nGMM Facies Distribution:")
print(df_clean['GMM_FACIES'].value_counts().sort_index())

# Set up our final prediction column
df_clean['DEN_RECONSTRUCTED'] = df_clean['DEN']

print("2. Training Facies-Specific Random Forest Models...")

# 2. Loop through each unique GMM Rock Type (0, 1, 2)
for facies in df_clean['GMM_FACIES'].unique():

    # Isolate data for THIS specific rock type
    df_facies_subset = df_clean[df_clean['GMM_FACIES'] == facies]

    # Split into Good Hole (Train) and Bad Hole (Predict/Reconstruct)
    train_data = df_facies_subset[df_facies_subset['WASHOUT'] == False]
    predict_data = df_facies_subset[df_facies_subset['WASHOUT'] == True]

    # Only train if we have both good data to learn from and bad data to fix
    if len(predict_data) > 0 and len(train_data) > 0:

        X_train = train_data[features]
        y_train = train_data[target]
        X_predict = predict_data[features]

        # Train a specific RF model JUST for this rock type
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)

        # Predict the missing Density for this specific rock type
        predicted_den = rf.predict(X_predict)

        # Inject the predictions back into the main dataframe
        df_clean.loc[predict_data.index, 'DEN_RECONSTRUCTED'] = predicted_den

        print(f"   Model trained and DEN reconstructed for GMM Facies {facies}")

print("All Facies-Specific Reconstructions Complete! Generating Final Plot...")

# 3. Plotting the Final Results
df_plot = df_clean[(df_clean['DEPTH'] > 3300) & (df_clean['DEPTH'] < 3600)].copy()
# ========== DIAGNOSTIC CODE - START ==========
print("\n" + "="*60)
print("       FACIES DISTRIBUTION DIAGNOSTIC")
print("="*60)

# Check entire well
print("\n1. ENTIRE WELL DISTRIBUTION:")
print(df_clean['GMM_FACIES'].value_counts().sort_index())
total_points = len(df_clean)
for facies_id in [0, 1, 2]:
    count = (df_clean['GMM_FACIES'] == facies_id).sum()
    percentage = (count / total_points) * 100
    if count > 0:
        facies_data = df_clean[df_clean['GMM_FACIES'] == facies_id]
        gr_mean = facies_data['GR'].mean()
        print(f"  Facies {facies_id}: {count:4d} points ({percentage:5.1f}%) - GR_mean={gr_mean:5.1f} API")
    else:
        print(f"  Facies {facies_id}: NOT PRESENT")

# Check plot range specifically
print(f"\n2. IN PLOT RANGE ({df_plot['DEPTH'].min():.0f}-{df_plot['DEPTH'].max():.0f}m):")
print(df_plot['GMM_FACIES'].value_counts().sort_index())
plot_points = len(df_plot)
for facies_id in [0, 1, 2]:
    count = (df_plot['GMM_FACIES'] == facies_id).sum()
    percentage = (count / plot_points) * 100
    if count > 0:
        facies_depths = df_plot[df_plot['GMM_FACIES'] == facies_id]['DEPTH']
        depth_min = facies_depths.min()
        depth_max = facies_depths.max()
        print(f"  Facies {facies_id}: {count:4d} points ({percentage:5.1f}%) - Depth: {depth_min:.1f}m to {depth_max:.1f}m")
    else:
        print(f"  Facies {facies_id}: ❌ NOT PRESENT IN THIS INTERVAL")

print("="*60 + "\n")
# ========== DIAGNOSTIC CODE - END ==========
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), sharey=True)
fig.suptitle('Facies-Specific AI Reconstruction (GMM + Random Forest)', fontsize=15, fontweight='bold', y=1.02)

# Track 1: Gamma Ray & Caliper Washout
ax[0].plot(df_plot['GR'], df_plot['DEPTH'], color='green', label='GR')
ax[0].fill_betweenx(df_plot['DEPTH'], 0, 150, where=df_plot['WASHOUT'], color='yellow', alpha=0.3, label='Washout Zone')
ax[0].set_xlabel("Gamma Ray (API)")
ax[0].set_xlim(0, 150)
ax[0].legend(loc='upper right')
ax[0].grid(True, linestyle='--', alpha=0.7)

# Track 2: GMM Facies Track
colors_facies = {
    0: '#DAA520',  # Facies 0: Clean Sand - Bright Gold
    1: '#FF8C00',  # Facies 1: Mixed Lithology - Dark Orange
    2: '#696969'   # Facies 2: Shale - Dim Gray
}

for facies_id, color in colors_facies.items():
    mask = (df_plot['GMM_FACIES'] == facies_id)
    ax[1].fill_betweenx(df_plot['DEPTH'], 0, 1,
                        where=mask,
                        facecolor=color,
                        alpha=0.95,
                        step='mid',
                        edgecolor='none')

ax[1].set_xlabel("GMM AI Rock Type")
ax[1].set_xlim(0, 1)
ax[1].set_xticks([])

# Add legend to explain the colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#DAA520', label='Clean Sand (0)'),
    Patch(facecolor='#FF8C00', label='Mixed Lithology (1)'),
    Patch(facecolor='#696969', label='Shale (2)')
]
ax[1].legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
# Track 3: The Reconstruction Overlay
ax[2].plot(df_plot['DEN'], df_plot['DEPTH'], color='red', linestyle='--', alpha=0.6, label='Raw DEN (Corrupted)')
ax[2].plot(df_plot['DEN_RECONSTRUCTED'], df_plot['DEPTH'], color='blue', linewidth=1.5, label='AI Reconstructed DEN')
ax[2].set_xlabel("Bulk Density (g/cc)")
ax[2].set_xlim(1.95, 2.95)
ax[2].invert_xaxis()
ax[2].legend(loc='upper left')
ax[2].grid(True, linestyle='--', alpha=0.7)

for a in ax:
    a.invert_yaxis()

plt.tight_layout()
plt.show()
