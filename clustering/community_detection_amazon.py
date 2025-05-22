#!/usr/bin/env python3
"""
Amazon Product User Community Analysis 

This script performs clustering analysis on Amazon datasets, currently Health & Household product review data, but can be changed to be Fashion. This is to identify user communities based on their purchasing and rating behavior to use for CoSiTi.
"""

import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from hdbscan import HDBSCAN


def safe_parse_array(arr):
    """
    Safely parse array-like strings or return the array if already parsed.
    
    Args:
        arr: String representation of array or actual array
        
    Returns:
        list: Parsed array or empty list if parsing fails
    """
    if isinstance(arr, str):
        try:
            return literal_eval(arr)
        except:
            return []
    return arr if isinstance(arr, list) else []


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the Amazon dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print("Loading and preprocessing data...")
    
    # Load data
    amazon = pd.read_csv(filepath)
    
    # Convert data types
    amazon["timestamp"] = pd.to_datetime(amazon["timestamp"], unit='ms')
    amazon["helpful_vote"] = pd.to_numeric(amazon["helpful_vote"], errors='coerce')
    amazon["verified_purchase"] = amazon["verified_purchase"].astype(bool)
    
    print(f"Loaded {len(amazon)} records")
    return amazon


def create_user_summary(amazon):
    """
    Create user summary statistics for clustering.
    
    Args:
        amazon (pd.DataFrame): Raw Amazon data - can be health & household or fashion
        
    Returns:
        pd.DataFrame: User summary statistics
    """
    print("Creating user summary statistics...")
    
    user_summary = amazon.groupby("userID").agg(
        unique_items_count=('itemID', 'nunique'),
        brands=('brand', lambda x: list(set(x.dropna()))),
        colours=('color', lambda x: list(set(x.dropna()))),
        titles=('title', lambda x: list(set(x))),
        category=('category', lambda x: list(set(x))),
        first_interaction_time=('timestamp', 'min'),
        last_interaction_time=('timestamp', 'max'),
        average_rating=('rating', lambda x: round(x.mean())),
        average_helpful_vote=('helpful_vote', lambda x: round(x.mean())),
        verified_purchase_count=('verified_purchase', lambda x: x.sum()) 
    ).reset_index()

    # Add derived features
    user_summary["brand_count"] = user_summary["brands"].apply(len)
    user_summary["category_count"] = user_summary["category"].apply(len)

    # Calculate recency
    latest_overall_time = amazon["timestamp"].max()
    user_summary["recency_days"] = (latest_overall_time - user_summary["last_interaction_time"]).dt.days
    
    print(f"Created summary for {len(user_summary)} users")
    return user_summary


def prepare_features(user_summary, n_features=50):
    """
    Prepare features for clustering by combining numerical and categorical features.
    
    Args:
        user_summary (pd.DataFrame): User summary data
        n_features (int): Number of features for hashing
        
    Returns:
        scipy.sparse.csr_matrix: Combined feature matrix
    """
    print("Preparing features for clustering...")
    
    # Prepare categorical arrays
    colour_arrays = user_summary['colours'].apply(safe_parse_array)
    category_arrays = user_summary['category'].apply(safe_parse_array)
    brand_arrays = user_summary['brands'].apply(safe_parse_array)

    # Build dicts for FeatureHasher
    colour_dicts = [{f"color_{x}": 1 for x in row} for row in colour_arrays]
    category_dicts = [{f"category_{x}": 1 for x in row} for row in category_arrays]
    brand_dicts = [{f"brand_{x}": 1 for x in row} for row in brand_arrays]

    # Initialize feature hashers
    colour_hasher = FeatureHasher(n_features=n_features, input_type='dict')
    category_hasher = FeatureHasher(n_features=n_features, input_type='dict')
    brand_hasher = FeatureHasher(n_features=n_features, input_type='dict')

    # Transform categorical features
    colour_features = colour_hasher.transform(colour_dicts)
    category_features = category_hasher.transform(category_dicts)
    brand_features = brand_hasher.transform(brand_dicts)

    # Define numerical columns
    numerical_cols = [
        'unique_items_count',
        'average_rating',
        'average_helpful_vote',
        'verified_purchase_count',
        'brand_count',
        'category_count',
        'recency_days'
    ]

    # Scale numerical features
    scaler = MinMaxScaler()
    scaled_numerical = scaler.fit_transform(user_summary[numerical_cols].fillna(0))
    numerical_features = csr_matrix(scaled_numerical)

    # Combine all features
    combined_features = hstack([
        numerical_features,
        colour_features,
        category_features,
        brand_features
    ]).tocsr()
    
    print(f"Prepared feature matrix of shape: {combined_features.shape}")
    return combined_features


def perform_dimensionality_reduction(combined_features):
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        combined_features: Feature matrix
        
    Returns:
        tuple: (2D UMAP embeddings, 2D UMAP embeddings for visualization)
    """
    print("Performing dimensionality reduction...")
    
    # UMAP for clustering
    reducer = UMAP(n_components=2, metric='jaccard', n_neighbors=25, random_state=42, min_dist=0.2)
    X_umap = reducer.fit_transform(combined_features)

    # UMAP for visualization
    reducer_2d = umap.UMAP(n_components=2, metric='jaccard', random_state=42, n_neighbors=25, min_dist=0.2) 
    X_umap_2d = reducer_2d.fit_transform(combined_features)
    
    return X_umap, X_umap_2d


def perform_clustering(X_umap):
    """
    Perform HDBSCAN clustering.
    
    Args:
        X_umap: UMAP embeddings
        
    Returns:
        np.array: Cluster labels
    """
    print("Performing clustering...")
    
    clusterer = HDBSCAN(min_cluster_size=30, min_samples=1, gen_min_span_tree=False, leaf_size=10) 
    labels = clusterer.fit_predict(X_umap)
    
    # Count clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Found {n_clusters} clusters with {n_noise} noise points")
    return labels


def visualize_clusters(X_umap_2d, labels, save_path=None):
    """
    Create visualization of clusters.
    
    Args:
        X_umap_2d: 2D UMAP embeddings for visualization
        labels: Cluster labels
        save_path (str, optional): Path to save the plot
    """
    print("Creating cluster visualization...")
    
    # Count occurrences of each label
    label_counts = pd.Series(labels).value_counts()
    
    if -1 in label_counts:
        label_counts = label_counts.drop(-1)

    top_clusters = label_counts.nlargest(6).index.tolist()
    all_top_clusters = list(set(top_clusters))
    unique_colors = sns.color_palette("tab20", n_colors=len(all_top_clusters))
    color_dict = {cluster: unique_colors[i] for i, cluster in enumerate(all_top_clusters)}

    mask = np.isin(labels, top_clusters)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_umap_2d[mask, 0], 
        y=X_umap_2d[mask, 1], 
        hue=labels[mask],
        palette=color_dict, 
        s=20, 
        alpha=0.7,
        legend=False
    )

    plt.xticks([])
    plt.yticks([])
    plt.title('Amazon Health & Household User Communities')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def evaluate_clustering(X_umap, labels):
    """
    Evaluate clustering quality using various metrics.
    
    Args:
        X_umap: UMAP embeddings
        labels: Cluster labels
    """
    print("\nEvaluating clustering quality...")
    
    # Only evaluate if we have more than one cluster (excluding noise)
    if len(set(labels)) - (1 if -1 in labels else 0) > 1:
        mask = labels != -1
        if sum(mask) > 0:  # Check if there are any non-noise points
            filtered_X_umap = X_umap[mask]
            filtered_labels = labels[mask]
            
            if len(set(filtered_labels)) > 1:  
                ch_score = calinski_harabasz_score(filtered_X_umap, filtered_labels)
                db_score = davies_bouldin_score(filtered_X_umap, filtered_labels)
                print(f"Calinski-Harabasz Score: {ch_score:.4f}")
                print(f"Davies-Bouldin Score: {db_score:.4f}")
                
                unique_labels, counts = np.unique(filtered_labels, return_counts=True)
                print("\nCluster sizes:")
                for label, count in zip(unique_labels, counts):
                    print(f"  Cluster {label}: {count} points")
            else:
                print("Only one cluster found after removing noise - cannot calculate metrics")


def analyze_communities(amazon, labels):
    """
    Perform qualitative analysis of identified communities.
    
    Args:
        amazon(pd.DataFrame): Original dataset
        labels: Cluster labels
    """
    print("\nAnalyzing community profiles...")
    
    # Add community labels to original data
    amazo = amazon.copy()
    
    # Create a mapping from userID to community
    user_summary_temp = amazon.groupby("userID").first().reset_index()
    user_summary_temp['Community'] = labels
    user_to_community = dict(zip(user_summary_temp['userID'], user_summary_temp['Community']))
    
    # Map communities to all records
    amazon['Community'] = amazon['userID'].map(user_to_community)
    
    # Qualitative analysis
    profile_features = ['color', 'brand', 'title']
    for community in sorted(amazon['Community'].unique()):
        if community == -1:  # Skip noise
            continue
            
        community_data = amazon[amazon['Community'] == community]
        print(f"\Community {community} Profil")
        print(f"Total records: {len(community_data)}")
        
        for feature in profile_features:
            if feature in community_data.columns:
                top_values = community_data[feature].value_counts().head(3)
                total = len(community_data)
                print(f"\nTop {feature}:")
                for val, count in top_values.items():
                    print(f"  {val}: {count} ({(count/total)*100:.1f}%)")
    
    return amazon


def save_results(amazon, user_summary, labels, output_path):
    """
    Save results to CSV file.
    
    Args:
        amazon (pd.DataFrame): Original dataset with communities
        user_summary (pd.DataFrame): User summary data
        labels: Cluster labels
        output_path (str): Path to save the results
    """
    print(f"Saving results to {output_path}...")
    
    # Add community labels to user summary
    user_summary['Community'] = labels
    
    label_counts = pd.Series(labels).value_counts()
    if -1 in label_counts:
        label_counts = label_counts.drop(-1)
    top_clusters = label_counts.nlargest(6).index.tolist()
    
    # Filter to only include top clusters
    df_top = user_summary[user_summary['Community'].isin(top_clusters)]
    df_top_subset = df_top[['userID', 'Community']]
    
    # Merge with original data
    merged_df = pd.merge(amazon, df_top_subset, on='userID', how='inner')
    merged_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(merged_df)} records from top {len(top_clusters)} communities")


def main():
    """
    Main function to run the complete analysis pipeline.
    """
    print("Starting Amazon Health & Household User Community Analysis")
    print("=" * 60)
    
    # Configuration
    input_file = 'amazon_health_household.csv'
    output_file = 'amazon_health_household_communities_data.csv'
    plot_file = 'amazon_fashion_cluster_paper.png'
    
    try:
        #  Load and preprocess data
        amazon= load_and_preprocess_data(input_file)
        
        # Create user summary
        user_summary = create_user_summary(amazon)
        
        # Prepare features
        combined_features = prepare_features(user_summary)
        
        # Dimensionality reduction
        X_umap, X_umap_2d = perform_dimensionality_reduction(combined_features)
        
        # Clustering
        labels = perform_clustering(X_umap)
        
        # Visualization
        visualize_clusters(X_umap_2d, labels, plot_file)
        
        # Evaluation
        evaluate_clustering(X_umap, labels)
        
        # Community analysis
        amazon_with_communities = analyze_communities(amazon, labels)
        
        # Save results 
        save_results(amazon_with_communities, user_summary, labels, output_file)
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()