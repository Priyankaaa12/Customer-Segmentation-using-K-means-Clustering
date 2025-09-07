import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Starting Customer Segmentation Analysis...")
    
    # Load the dataset
    try:
        df = pd.read_csv('Mall_Customers.csv')
        print(" Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(" Error: File 'Mall_Customers.csv' not found!")
        print("Please make sure the file is in the same folder as this script.")
        return
    
    # Data preprocessing
    print("\nPreprocessing data...")
    df_clean = df.dropna()
    df_clean['Gender'] = df_clean['Gender'].map({'Male': 0, 'Female': 1})
    
    # Select features and scale
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df_clean[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters using Elbow method
    print(" Finding optimal number of clusters...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Square)')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Elbow method plot saved as 'elbow_method.png'")
    
    # Apply K-means with optimal clusters
    optimal_k = 5
    print(f" Applying K-means with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['Cluster'] = clusters
    
    # Analyze clusters
    print("\n Analyzing clusters...")
    cluster_summary = df_clean.groupby('Cluster')[features].mean()
    print("Cluster Summary Statistics:")
    print(cluster_summary)
    
    # Save results
    print("\n Saving results...")
    df_clean.to_csv('customer_segmentation_results.csv', index=False)
    print(" Results saved as 'customer_segmentation_results.csv'")
    
    # Create PCA visualization
    print(" Creating cluster visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Customer Segments Visualization (PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Cluster visualization saved as 'pca_clusters.png'")
    
    # === KEY CUSTOMER INSIGHTS === 
    print("\n=== KEY CUSTOMER INSIGHTS ===")
    
    # Find your most valuable customers (VIPs) - Cluster 2
    vip_customers = df_clean[df_clean['Cluster'] == 2]
    print(f" VIP CUSTOMERS: {len(vip_customers)} people")
    print(f"   Average income: ${vip_customers['Annual Income (k$)'].mean():.1f}k")
    print(f"   Average spending: {vip_customers['Spending Score (1-100)'].mean():.1f}/100")
    
    # Find customers with potential - Cluster 1
    potential_customers = df_clean[df_clean['Cluster'] == 1]
    print(f"\n POTENTIAL CUSTOMERS: {len(potential_customers)} people")
    print(f"   Average income: ${potential_customers['Annual Income (k$)'].mean():.1f}k")
    print(f"   Average spending: {potential_customers['Spending Score (1-100)'].mean():.1f}/100")
    print("    Opportunity: High income but low spending - target with special offers!")
    
    # Generate detailed report
    print("\n" + "="*60)
    print(" DETAILED CLUSTER ANALYSIS REPORT")
    print("="*60)
    
    cluster_names = {
        0: "Standard Customers",
        1: "Careful Spenders", 
        2: "Target Customers (VIP)",
        3: "Young Big Spenders",
        4: "Conservative Customers"
    }
    
    for cluster_id in range(optimal_k):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        print(f"\n {cluster_names[cluster_id]} (Cluster {cluster_id})")
        print(f"   Size: {len(cluster_data)} customers ({len(cluster_data)/len(df_clean)*100:.1f}%)")
        print(f"   Avg Age: {cluster_data['Age'].mean():.1f} years")
        print(f"   Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
        print(f"    Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}/100")
        
        # Gender distribution
        female_percent = cluster_data['Gender'].mean() * 100
        print(f"    Female: {female_percent:.1f}% |  Male: {100-female_percent:.1f}%")
        
        # Business recommendations
        if cluster_id == 2:  # Target Customers
            print("    Recommendation: Premium loyalty program, exclusive offers")
        elif cluster_id == 1:  # Careful Spenders
            print("    Recommendation: Discount campaigns, value-based marketing")
        elif cluster_id == 3:  # Young Big Spenders
            print("    Recommendation: Trendy products, social media marketing")
        elif cluster_id == 4:  # Conservative Customers
            print("    Recommendation: Budget-friendly options, essential products")
        else:
            print("    Recommendation: Standard marketing campaigns")
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE!")
    print("="*60)
    print(" Files created:")
    print("   - customer_segmentation_results.csv")
    print("   - elbow_method.png") 
    print("   - pca_clusters.png")
    
    # Return the cleaned dataframe for further analysis if needed
    return df_clean

if __name__ == "__main__":
    df_clean = main()
    
    # Now you can do additional analysis here if needed
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSIS")
    print("="*50)
    
    # Example: Show cluster distribution
    print("Cluster Distribution:")
    print(df_clean['Cluster'].value_counts().sort_index())
    
    # Example: Find highest spending customers
    top_spenders = df_clean.nlargest(5, 'Spending Score (1-100)')
    print(f"\n Top 5 Highest Spending Customers:")
    for i, row in top_spenders.iterrows():
        print(f"   Customer: {row['CustomerID'] if 'CustomerID' in row else 'N/A'}, "
              f"Spending: {row['Spending Score (1-100)']}, "
              f"Cluster: {row['Cluster']}")