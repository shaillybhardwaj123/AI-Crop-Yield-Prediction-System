import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Project directories setup
print("=== Step 1: Initializing Project Directories ===")
dirs = ["data", "notebooks", "images", "models", "app"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Directory ensured: {d}/")

# 2. Copying dataset from Downloads if not in data/
csv_dest = os.path.join("data", "yield_df.csv")
csv_src = r"c:\Users\hp\Downloads\yield_df.csv"

# Copying generated banner to images/ if it exists
banner_src = r"C:\Users\hp\.gemini\antigravity-ide\brain\773fee94-89b2-494c-ae3a-0b9d5b72452e\banner_1780663061240.png"
banner_dest = os.path.join("images", "banner.png")
if os.path.exists(banner_src) and not os.path.exists(banner_dest):
    try:
        shutil.copy(banner_src, banner_dest)
        print("Repository banner successfully copied to images/banner.png")
    except Exception as e:
        pass

if not os.path.exists(csv_dest):
    print("\n=== Step 2: Locating Dataset ===")
    if os.path.exists(csv_src):
        try:
            print(f"Copying yield_df.csv from Downloads: {csv_src} -> {csv_dest}")
            shutil.copy(csv_src, csv_dest)
            print("Dataset successfully copied to project data/ folder.")
        except Exception as e:
            print(f"Error copying dataset: {e}. Please copy the file manually.")
    else:
        print(f"Warning: Could not find dataset at {csv_src} or {csv_dest}.")
        print("Please manually place 'yield_df.csv' in the 'data/' directory.")

# 3. Load and Overview Dataset
if os.path.exists(csv_dest):
    print("\n=== Step 3: Loading and Cleaning Data ===")
    df = pd.read_csv(csv_dest)
    
    # Remove unnamed index column if it exists
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    
    print("Dataset dimensions:", df.shape)
    print("Columns in dataset:", list(df.columns))
    
    # Clean column names for easier access (optional but good practice)
    # df.columns = df.columns.str.strip()
    
    # Missing Values check
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Drop duplicates if any
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Removing {duplicates_count} duplicate rows.")
        df = df.drop_duplicates()
        
    print(df.describe())
    
    # 4. Exploratory Data Analysis & Visualizations
    print("\n=== Step 4: Generating Exploratory Data Analysis & Visualizations ===")
    sns.set_theme(style="whitegrid")
    
    # Visualization 1: Yield Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['hg/ha_yield'], kde=True, color='#2ec4b6', bins=30)
    plt.title('Distribution of Crop Yield (hg/ha)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Yield (hg/ha)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'yield_distribution.png'), dpi=300)
    plt.close()
    print("Saved: images/yield_distribution.png")

    # Visualization 2: Rainfall vs Yield Plot (Sampled for visual clarity)
    plt.figure(figsize=(10, 6))
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    sns.scatterplot(data=df_sample, x='average_rain_fall_mm_per_year', y='hg/ha_yield', alpha=0.5, color='#457b9d')
    plt.title('Rainfall vs Crop Yield (Sampled)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Rainfall (mm/year)', fontsize=12)
    plt.ylabel('Yield (hg/ha)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'rainfall_vs_yield.png'), dpi=300)
    plt.close()
    print("Saved: images/rainfall_vs_yield.png")

    # Visualization 3: Temperature vs Yield Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sample, x='avg_temp', y='hg/ha_yield', alpha=0.5, color='#e63946')
    plt.title('Temperature vs Crop Yield (Sampled)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Temperature (°C)', fontsize=12)
    plt.ylabel('Yield (hg/ha)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'temperature_vs_yield.png'), dpi=300)
    plt.close()
    print("Saved: images/temperature_vs_yield.png")

    # Visualization 4: Top Crop Producing Areas
    plt.figure(figsize=(12, 6))
    top_areas = df.groupby('Area')['hg/ha_yield'].mean().nlargest(10).reset_index()
    sns.barplot(data=top_areas, x='hg/ha_yield', y='Area', palette='viridis')
    plt.title('Top 10 Crop Producing Countries (Average Yield)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Yield (hg/ha)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'top_crops.png'), dpi=300)
    plt.close()
    print("Saved: images/top_crops.png")

    # Visualization 5: Correlation Heatmap (numerical features)
    plt.figure(figsize=(8, 6))
    numerical_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'), dpi=300)
    plt.close()
    print("Saved: images/heatmap.png")

    # 5. Preprocessing & Encoding Variables
    print("\n=== Step 5: Preprocessing and Train-Test Split ===")
    X = df.drop(columns=['hg/ha_yield'])
    y = df['hg/ha_yield']
    
    # Categorical and numerical columns
    categorical_features = ['Area', 'Item']
    numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    # Save the unique items and areas for our streamlit app selection
    unique_areas = sorted(df['Area'].unique().tolist())
    unique_items = sorted(df['Item'].unique().tolist())
    
    # Define Column Transformer for pipeline preprocessing with version safety for sparse/sparse_output
    try:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', cat_encoder, categorical_features)
        ])
    
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 6. Model Training & Evaluation
    print("\n=== Step 6: Model Training and Evaluation ===")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=15, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    }
    
    results = {}
    fitted_pipelines = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        fitted_pipelines[name] = pipeline
        
        print(f"  -> R² Score: {r2:.4f}")
        print(f"  -> MAE: {mae:.2f}")
        print(f"  -> RMSE: {rmse:.2f}")

    # Visualization 6: Prediction Comparison Graph
    plt.figure(figsize=(10, 6))
    rf_preds = fitted_pipelines['Random Forest Regressor'].predict(X_test)
    dt_preds = fitted_pipelines['Decision Tree Regressor'].predict(X_test)
    lr_preds = fitted_pipelines['Linear Regression'].predict(X_test)
    
    # Plotting first 50 values for clarity
    plt.plot(np.array(y_test)[:50], label='Actual Yield', color='black', linewidth=2.5, marker='o')
    plt.plot(rf_preds[:50], label='Random Forest Preds', color='#2ec4b6', linestyle='--', marker='x')
    plt.plot(dt_preds[:50], label='Decision Tree Preds', color='#ff9f1c', linestyle=':', marker='^')
    plt.plot(lr_preds[:50], label='Linear Regression Preds', color='#e63946', linestyle='-.', marker='s')
    
    plt.title('Actual vs Predicted Yield Comparison (First 50 Samples)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Yield (hg/ha)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'prediction_comparison.png'), dpi=300)
    plt.close()
    print("Saved: images/prediction_comparison.png")

    # Feature Importance (Random Forest Regressor)
    print("\n=== Step 7: Generating Feature Importance Plot ===")
    rf_pipeline = fitted_pipelines['Random Forest Regressor']
    
    # Extract feature names after transformer
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    one_hot_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + one_hot_cols
    
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    
    # Group importance for categorical variables back to unique categories to make a clean plot
    # Or just plot top 15 features
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='mako')
    plt.title('Top 15 Feature Importances (Random Forest Regressor)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Relative Importance Value', fontsize=12)
    plt.ylabel('Features / One-Hot Encoding Categories', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'feature_importance.png'), dpi=300)
    plt.close()
    print("Saved: images/feature_importance.png")

    # 7. Model Saving using pickle
    print("\n=== Step 8: Serializing Models ===")
    
    # Save primary model pipeline
    with open(os.path.join('models', 'random_forest.pkl'), 'wb') as f:
        pickle.dump(fitted_pipelines['Random Forest Regressor'], f)
    print("Saved primary Random Forest model: models/random_forest.pkl")

    # Save additional models
    with open(os.path.join('models', 'decision_tree.pkl'), 'wb') as f:
        pickle.dump(fitted_pipelines['Decision Tree Regressor'], f)
    print("Saved Decision Tree model: models/decision_tree.pkl")

    with open(os.path.join('models', 'linear_regression.pkl'), 'wb') as f:
        pickle.dump(fitted_pipelines['Linear Regression'], f)
    print("Saved Linear Regression model: models/linear_regression.pkl")
    
    # Save application metadata (lists of areas and crop items)
    app_meta = {
        'areas': unique_areas,
        'items': unique_items,
        'results': results
    }
    with open(os.path.join('models', 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(app_meta, f)
    print("Saved App Preprocessor Metadata: models/preprocessor.pkl")
    
    print("\n=== pipeline execution complete. All assets, models, and visuals have been saved! ===")
else:
    print("\nCould not proceed: Dataset not found in data/yield_df.csv")
