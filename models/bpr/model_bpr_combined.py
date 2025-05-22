import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import recall_at_k
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict

np.random.seed(42)

# LOAD & PREPROCESS DATA 
file_path = "data/amazon_health_household_commmunities.csv"
usecols = ["userID", "itemID", "timestamp", "Community", 'brand', 'category']
df = pd.read_csv(file_path, sep=",", usecols=usecols, low_memory=False)

df['event_date'] = pd.to_datetime(df['timestamp'])
most_recent_date = df['event_date'].max()
df['days_since'] = (most_recent_date - df['event_date']).dt.days

print(f"Data loaded: {len(df)} interactions, {df['userID'].nunique()} users, {df['itemID'].nunique()} items")

#  MAP USER & ITEM IDS 
user_map = {user: u for u, user in enumerate(df['userID'].unique())}
item_map = {item: i for i, item in enumerate(df['itemID'].unique())}

df["userID"] = df["userID"].map(user_map)
df["itemID"] = df["itemID"].map(item_map)
n_users = len(user_map)
n_items = len(item_map)

user_community = df.groupby('userID')['Community'].first().to_dict()
item_community = df.groupby('itemID')['Community'].first().to_dict()

print(f"Mapped {n_users} users and {n_items} items")

# TRAIN-TEST SPLIT
df["rank"] = df.groupby("userID").cumcount(ascending=False) + 1
train_df = df[df["rank"] > 1].drop(columns=["rank"])
test_df = df[df["rank"] == 1].drop(columns=["rank"])
print(f"Train set: {len(train_df)} interactions, Test set: {len(test_df)} interactions")

# CONVERT TO INTERACTION MATRICES 
dataset = Dataset()
dataset.fit(range(n_users), range(n_items))

(interactions_train, _) = dataset.build_interactions([
    (row.userID, row.itemID) for _, row in train_df.iterrows()
])
(interactions_test, _) = dataset.build_interactions([
    (row.userID, row.itemID) for _, row in test_df.iterrows()
])

#  CALCULATE ITEM SIMILARITY MATRIX 
item_features_df = df[['itemID', 'category', 'brand', 'Community']].drop_duplicates()

item_community_counts = df.groupby(['itemID', 'Community']).size().reset_index(name='count')
item_primary_community = item_community_counts.sort_values(['itemID', 'count'], ascending=[True, False])
item_primary_community = item_primary_community.drop_duplicates('itemID')

item_features_df = pd.merge(
    item_features_df[['itemID', 'category', 'brand']],
    item_primary_community[['itemID', 'Community']],
    on='itemID',
    how='left'
)

hfb_dummies = pd.get_dummies(item_features_df['category'], prefix='hfb')
series_dummies = pd.get_dummies(item_features_df['brand'], prefix='series')
community_dummies = pd.get_dummies(item_features_df['Community'], prefix='community')

item_features = pd.concat([hfb_dummies, series_dummies, community_dummies], axis=1)

item_similarity_matrix = cosine_similarity(item_features.to_numpy())
print(f"Created item similarity matrix of shape {item_similarity_matrix.shape}")

item_to_community = dict(zip(item_features_df['itemID'], item_features_df['Community']))

communities = item_features_df['Community'].unique()
items_by_community = {}
for comm in communities:
    items_by_community[comm] = item_features_df[item_features_df['Community'] == comm]['itemID'].values

print(f"Prepared community mappings for {len(communities)} communities")

# CREATE USER-ITEM-TIMESTAMP DICTIONARY 
print("Building user-item-timestamp dictionary...")
user_item_timestamps = {}
for _, row in train_df.iterrows():
    try:
        user_id = row['userID']
        item_id = row['itemID']
        timestamp = row['event_date'].timestamp()  
        
        if user_id not in user_item_timestamps:
            user_item_timestamps[user_id] = {}
        user_item_timestamps[user_id][item_id] = timestamp
    except (ValueError, AttributeError) as e:
        continue

train_interactions = list(zip(train_df['userID'], train_df['itemID']))
test_interactions = list(zip(test_df['userID'], test_df['itemID']))

# NORMALIZE ITEM SIMILARITY MATRIX 
def normalize_item_similarity_matrix(item_similarity_matrix):
    """
    Normalize the entire item similarity matrix to ensure pairwise similarities 
    have consistent scale. This preserves the relative similarity between specific pairs
    of items, rather than using a global summary statistic per item.
    
    Returns:
        normalized_matrix: A normalized version of the similarity matrix with values in [0,1]
    """
    normalized_matrix = item_similarity_matrix.copy()
    
    min_sim = np.min(normalized_matrix)
    max_sim = np.max(normalized_matrix)
    
    if max_sim > min_sim:
        normalized_matrix = (normalized_matrix - min_sim) / (max_sim - min_sim)
    
    return normalized_matrix

#  TIME-BASED COMMUNITY SIMILARITY FUNCTION 
def calculate_time_similarity(user_id, item1, item2, user_item_timestamps, user_community, item_community):
    """
    Calculate time-based similarity between two items for a user, but only if they're in the same community.
    Returns a value between 0-1 where higher values indicate items consumed closer in time.
    """
    if item_community.get(item1) != item_community.get(item2):
        return 0.0
        
    if user_id not in user_item_timestamps or \
       item1 not in user_item_timestamps[user_id] or \
       item2 not in user_item_timestamps[user_id]:
        return 0.0
    
    time_diff = abs(user_item_timestamps[user_id][item1] - user_item_timestamps[user_id][item2])
    
    time_diff_days = time_diff / (24 * 60 * 60)

    similarity = np.exp(-time_diff_days / 7)
    
    return similarity

# COMBINED RECOMMENDATIONS FUNCTION 
def combined_recommendations(model, user_id, item_ids, item_similarity_matrix, 
                           user_item_timestamps, user_community, item_community,
                           normalized_item_sim_matrix, alpha_bpr=0.7, alpha_item=0.2, alpha_time=0.1):
    """
    Apply community-based time similarity and item similarity to BPR scores with bias correction.
    
    Parameters:
    -----------
    alpha_bpr : float
        Weight for BPR model predictions
    alpha_item : float
        Weight for item similarity scores
    alpha_time : float
        Weight for time decay scores (community-only)
    """
    bpr_scores = np.zeros(len(item_ids))
    for i, item_id in enumerate(item_ids):
        bpr_scores[i] = model.predict(np.array([user_id]), np.array([item_id]))[0]
    
    item_sim_scores = np.zeros_like(bpr_scores)
    for i, item_id in enumerate(item_ids):
        if item_id < normalized_item_sim_matrix.shape[0]:
            item_sim_scores[i] = np.mean(normalized_item_sim_matrix[item_id])
    
    time_scores = np.zeros_like(bpr_scores)
    
    if user_id in user_item_timestamps:
        user_items = list(user_item_timestamps[user_id].keys())
        
        for i, item_id in enumerate(item_ids):
            same_community_items = [
                item for item in user_items 
                if item_community.get(item) == item_community.get(item_id)
            ]
            
            if same_community_items:
                similarities = []
                for past_item in same_community_items:
                    sim = calculate_time_similarity(user_id, past_item, item_id, 
                                                 user_item_timestamps, 
                                                 user_community, item_community)
                    similarities.append(sim)
                
                if similarities:
                    time_scores[i] = np.mean(similarities)
    
    for scores in [bpr_scores, item_sim_scores, time_scores]:
        score_range = np.max(scores) - np.min(scores)
        if score_range > 1e-10:
            scores[:] = (scores - np.min(scores)) / score_range
        else:
            scores[:] = np.zeros_like(scores)
    
    corr_bpr_item = 0
    corr_bpr_time = 0
    corr_item_time = 0
    
    if np.std(bpr_scores) > 1e-10 and np.std(item_sim_scores) > 1e-10:
        corr_bpr_item = np.corrcoef(bpr_scores, item_sim_scores)[0,1]
        if np.isnan(corr_bpr_item):
            corr_bpr_item = 0
            
    if np.std(bpr_scores) > 1e-10 and np.std(time_scores) > 1e-10:
        corr_bpr_time = np.corrcoef(bpr_scores, time_scores)[0,1]
        if np.isnan(corr_bpr_time):
            corr_bpr_time = 0
            
    if np.std(item_sim_scores) > 1e-10 and np.std(time_scores) > 1e-10:
        corr_item_time = np.corrcoef(item_sim_scores, time_scores)[0,1]
        if np.isnan(corr_item_time):
            corr_item_time = 0

    correction_item = 1.0 - abs(corr_bpr_item)
    correction_time = 1.0 - abs(corr_bpr_time)
    
    redundancy_factor = 1.0 - abs(corr_item_time) * 0.5  
    
    effective_item_weight = alpha_item * correction_item * redundancy_factor
    effective_time_weight = alpha_time * correction_time * redundancy_factor
    
    effective_bpr_weight = 1.0 - effective_item_weight - effective_time_weight
    
    combined_scores = (
        effective_bpr_weight * bpr_scores + 
        effective_item_weight * item_sim_scores + 
        effective_time_weight * time_scores
    )
    
    return combined_scores

# EVALUATION FUNCTION 
def evaluate_hybrid_model(model, test_interactions, train_interactions, 
                          item_similarity_matrix, user_item_timestamps, 
                          user_community, item_community, normalized_item_sim_matrix,
                          alpha_bpr=0.4, alpha_item=0.3, alpha_time=0.3,
                          ks=[5, 10, 20]):
    """
    Evaluates the hybrid recommendation model combining BPR, item similarity, and community-based time similarity.
    
    Parameters:
    -----------
    model : LightFM model
        The trained recommendation model
    test_interactions : list of tuples
        List of (user_id, item_id) tuples for testing
    train_interactions : list of tuples
        List of (user_id, item_id) tuples for training
    item_similarity_matrix : numpy.ndarray
        Matrix of item similarities
    user_item_timestamps : dict
        Dictionary containing timestamps for user-item interactions
    user_community : dict
        Dictionary mapping users to their communities
    item_community : dict
        Dictionary mapping items to their communities
    normalized_item_sim_matrix : numpy.ndarray
        Normalized version of item similarity matrix
    alpha_bpr : float, default=0.4
        Weight for BPR component
    alpha_item : float, default=0.3
        Weight for item similarity component
    alpha_time : float, default=0.3
        Weight for time similarity component
    ks : list of int
        List of cutoff values to evaluate (e.g., [5, 10, 20])
        
    Returns:
    --------
    dict
        Dictionary with metrics for each k
    """
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    debug_stats = {'users_skipped': 0, 'users_evaluated': 0, 'no_diversity': 0}
    max_sim_items = len(item_similarity_matrix)
    total_evaluated = 0
    
    for u, i in train_interactions:
        user_interactions[u].add(i)
    
    all_items = set(range(n_items))
    
    for user_idx, (u, true_i) in enumerate(test_interactions):
        if u not in user_interactions:
            debug_stats['users_skipped'] += 1
            continue
            
        seen = user_interactions[u]
        
        candidates = list(all_items - seen)
        
        if len(candidates) < 99:
            additional_candidates = list(seen - {true_i})
            candidates = (candidates + additional_candidates)[:99]
            if len(candidates) < 99:
                debug_stats['users_skipped'] += 1
                continue
        else:
            np.random.shuffle(candidates)
            candidates = candidates[:99]
        
        test_items = candidates + [true_i]
        
        hybrid_scores = combined_recommendations(
            model, u, test_items, item_similarity_matrix, 
            user_item_timestamps, user_community, item_community,
            normalized_item_sim_matrix, alpha_bpr, alpha_item, alpha_time
        )
        
        top_k_indices = np.argsort(hybrid_scores)[::-1]
        top_k_items = [test_items[i] for i in top_k_indices]
        
        for k in ks:
            actual_k = min(k, len(top_k_items))
            top_k = top_k_items[:actual_k]
            
            hit = 1 if true_i in top_k else 0
            metrics[k]['hit'] += hit
            
            if hit > 0:
                rank = top_k.index(true_i) + 1
                modified_recall = 1.0 / rank  
                metrics[k]['recall'] += modified_recall
            
            if hit > 0:
                binary_relevance = [1 if i == true_i else 0 for i in top_k]
                ideal = [1] + [0] * (len(binary_relevance) - 1)  
                try:
                    metrics[k]['ndcg'] += ndcg_score(np.array([ideal]), np.array([binary_relevance]))
                except Exception as e:
                    pass
        
        debug_stats['users_evaluated'] += 1
        total_evaluated += 1
    
    results = {}
    for k in ks:
        results[k] = {
            'recall': metrics[k]['recall'] / total_evaluated,
            'ndcg': metrics[k]['ndcg'] / total_evaluated,
            'hit_rate': metrics[k]['hit'] / total_evaluated
        }
    
    return results


# MAIN EXECUTION
if __name__ == "__main__":
    alpha_item_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    alpha_time_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]

    alpha_combinations = []
    for alpha_item in alpha_item_values:
        for alpha_time in alpha_time_values:
            alpha_bpr = 1.0 - alpha_item - alpha_time
            if alpha_bpr >= 0.0:  
                alpha_combinations.append((alpha_bpr, alpha_item, alpha_time))
    
    alpha_combinations.sort(key=lambda x: x[0], reverse=True)

    normalized_item_sim_matrix = normalize_item_similarity_matrix(item_similarity_matrix)

    num_runs = 3
    ks = [5, 10, 20]
    
    print(f"Starting grid search with {len(alpha_combinations)} alpha combinations, {num_runs} runs each")

    grid_search_results = []
    all_runs_results = [] 
    start_time = time.time()
    
    for combo_idx, (alpha_bpr, alpha_item, alpha_time) in enumerate(tqdm(alpha_combinations, desc="Testing alpha combinations")):
        print(f"\nCombination {combo_idx+1}/{len(alpha_combinations)}: bpr={alpha_bpr:.2f}, item={alpha_item:.2f}, time={alpha_time:.2f}")
        
        aggregated_results = {k: {metric: 0.0 for metric in ['recall', 'ndcg', 'hit_rate']} for k in ks}
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}")
            
            np.random.seed(42 + run)
            
            model = LightFM(loss="bpr", no_components=64, learning_rate=0.01, random_state=42 + run)
            model.fit(interactions_train, epochs=100, num_threads=8, verbose=False)
    
            try:
                run_results = evaluate_hybrid_model(
                    model, 
                    test_interactions, 
                    train_interactions,
                    item_similarity_matrix, 
                    user_item_timestamps,
                    user_community,
                    item_community,
                    normalized_item_sim_matrix,
                    alpha_bpr=alpha_bpr,
                    alpha_item=alpha_item,
                    alpha_time=alpha_time,
                    ks=ks
                )
                
                run_data = {
                    'alpha_bpr': alpha_bpr,
                    'alpha_item': alpha_item,
                    'alpha_time': alpha_time,
                    'run': run
                }
                
                for k in ks:
                    for metric in ['recall', 'ndcg', 'hit_rate']:
                        run_data[f"{metric}@{k}"] = run_results[k][metric]
                
                all_runs_results.append(run_data)
                
                for k in ks:
                    for metric in ['recall', 'ndcg', 'hit_rate']:
                        aggregated_results[k][metric] += run_results[k][metric] / num_runs
                
                info = f"    Results: "
                for k in ks:
                    info += f"R@{k}={run_results[k]['recall']:.4f}, "
                    info += f"NDCG@{k}={run_results[k]['ndcg']:.4f}, "
                    info += f"HR@{k}={run_results[k]['hit_rate']:.4f} | "
                print(info)
            except Exception as e:
                print(f"  Error in evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        result_row = {
            'alpha_bpr': alpha_bpr,
            'alpha_item': alpha_item,
            'alpha_time': alpha_time
        }
        
        for k in ks:
            for metric in ['recall', 'ndcg', 'hit_rate']:
                result_row[f"{metric}@{k}"] = aggregated_results[k][metric]
        
        grid_search_results.append(result_row)
        
        print(f"  Average results for bpr={alpha_bpr:.2f}, item={alpha_item:.2f}, time={alpha_time:.2f}:")
        for k in ks:
            print(f"    @K={k} → " +
                  f"Recall: {aggregated_results[k]['recall']:.4f}, " +
                  f"NDCG: {aggregated_results[k]['ndcg']:.4f}, " +
                  f"Hit Rate: {aggregated_results[k]['hit_rate']:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame(grid_search_results)
    csv_file = f"bpr_community_time_grid_search_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    
    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_csv = f"bpr_community_time_grid_search_all_runs_{timestamp}.csv"
    all_runs_df.to_csv(all_runs_csv, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nGrid search complete in {elapsed_time/60:.2f} minutes.")
    print(f"Results saved to:")
    print(f"- Aggregated results: '{csv_file}'")
    print(f"- All individual runs: '{all_runs_csv}'")