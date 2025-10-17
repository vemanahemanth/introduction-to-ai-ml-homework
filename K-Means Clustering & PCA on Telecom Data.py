import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Load Data ---
# !!! UPDATE THIS PATH !!!
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 
try:
    df = pd.read_csv(file_path)
    print("Telco Churn dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please download the dataset from Kaggle and update the file_path variable.")
    # Exit or raise error, but for this example, we'll stop
    # In a real script, you'd exit here.
    df = None 

if df is not None:
    
    # --- Part A: Preprocess data ---
    print("\n--- Part A: Preprocessing ---")
    
    # Drop customerID
    df_clean = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, coercing errors (empty strings) to NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    # Drop rows with NaN values (they are a small fraction)
    df_clean = df_clean.dropna()

    # Keep original df for Part D analysis
    df_original_clean = df_clean.copy()
    
    # Drop Churn for unsupervised learning
    df_cluster = df_clean.drop('Churn', axis=1)

    # Identify numeric and categorical features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # 'SeniorCitizen' is 0/1, we'll treat it as categorical
    categorical_features = df_cluster.select_dtypes(include=['object']).columns.tolist()
    categorical_features.append('SeniorCitizen')

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create the full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(df_cluster)
    print(f"Data preprocessed. New shape: {X_preprocessed.shape}")


    # --- Part B: PCA to 2 components ---
    print("\n--- Part B: PCA ---")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_preprocessed)

    # Q: How much variance is explained?
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by PC1: {explained_variance[0]:.2%}")
    print(f"Variance explained by PC2: {explained_variance[1]:.2%}")
    print(f"Total variance explained by 2 components: {np.sum(explained_variance):.2%}")

    # Plot 2D scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10)
    plt.title('PCA 2-Component Scatter Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    print("Q: Any visible groups? (A: Usually not very clear, maybe one large blob.)")


    # --- Part C: K-Means on PCA data ---
    print("\n--- Part C: K-Means ---")
    
    # Use Elbow & Silhouette to find K
    k_range = range(2, 9)
    inertia_values = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_pca)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

    # Plot Elbow & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertia_values, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    plt.show()

    # Q: What K was chosen?
    # A: Look at the plots. The elbow is often at k=3 or k=4.
    #    The silhouette score might also peak around 3 or 4.
    K_CHOSEN = 3  # <-- **** YOU MIGHT NEED TO CHANGE THIS BASED ON YOUR PLOTS ****
    print(f"Q: What K was chosen? (A: Based on plots, K={K_CHOSEN} seems reasonable.)")

    # Run K-Means with chosen K
    kmeans = KMeans(n_clusters=K_CHOSEN, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # Plot clusters
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=20)
    plt.title(f'K-Means Clustering (k={K_CHOSEN}) on PCA Data')
    plt.show()
    
    print("Q: Do clusters make business sense? (A: See Part D analysis.)")


    # --- Part D (Bonus): Analyze clusters ---
    print(f"\n--- Part D: Cluster Analysis (k={K_CHOSEN}) ---")
    
    # Add cluster labels back to the ORIGINAL (clean) dataframe
    df_original_clean['Cluster'] = cluster_labels
    
    # Q: Describe segments and telecom insights.
    
    # Analyze by numeric features
    print("Average Numeric Values by Cluster:")
    numeric_analysis = df_original_clean.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean()
    print(numeric_analysis)
    
    # Analyze by Churn rate
    print("\nChurn Rate by Cluster:")
    churn_analysis = df_original_clean.groupby('Cluster')['Churn'].value_counts(normalize=True).unstack()
    churn_analysis['Churn_Rate_%'] = churn_analysis.get('Yes', 0) * 100
    print(churn_analysis[['Churn_Rate_%']])
    
    # Analyze by a categorical feature, e.g., Contract
    print("\nContract Type by Cluster (Normalized):")
    contract_analysis = df_original_clean.groupby('Cluster')['Contract'].value_counts(normalize=True).unstack().fillna(0)
    print(contract_analysis)

    print("\nExample Insights (Answers):")
    print(" * Based on the analysis, you can create 'personas'.")
    print(" * e.g., Cluster 0 might be 'New, High-Churn Customers' (low tenure, month-to-month, high churn).")
    print(" * e.g., Cluster 1 might be 'Loyal Veterans' (high tenure, long-term contracts, low churn).")
    print(" * e.g., Cluster 2 might be 'Mid-Term Customers' (medium tenure, mixed contracts).")
    print(" * Insight: The business can target Cluster 0 with retention offers.")