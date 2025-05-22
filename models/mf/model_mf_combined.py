import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix
import time
from datetime import datetime
from collections import defaultdict

np.random.seed(42)

# LOAD & PREPROCESS DATA
file_path = "data/amazon_health_household_communities.csv"
usecols = ["userID", "itemID", "timestamp", "Community", 'brand', 'category']
df = pd.read_csv(file_path, sep=",", usecols=usecols, low_memory=False)

df['event_date'] = pd.to_datetime(df['timestamp'])
most_recent_date = df['event_date'].max()
df['days_since'] = (most_recent_date - df['event_date']).dt.days

print(f"Data loaded: {len(df)} interactions, {df['userID'].nunique()} users, {df['itemID'].nunique()} items")

# MAP USER & ITEM IDs
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

train_interactions = list(zip(train_df['userID'], train_df['itemID']))
test_interactions = list(zip(test_df['userID'], test_df['itemID']))

#  BUILD INTERACTION MATRICES 
train_matrix = csr_matrix((np.ones(len(train_df)), (train_df["userID"], train_df["itemID"])), shape=(n_users, n_items))

#COMPUTE ITEM SIMILARITY MATRIX
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

item_features_matrix = item_features.to_numpy()
item_similarity_matrix = cosine_similarity(item_features_matrix)
print(f"Created item similarity matrix of shape {item_similarity_matrix.shape}")

item_to_community = dict(zip(item_features_df['itemID'], item_features_df['Community']))

communities = item_features_df['Community'].unique()
items_by_community = {}
for comm in communities:
    items_by_community[comm] = item_features_df[item_features_df['Community'] == comm]['itemID'].values

print(f"Prepared community mappings for {len(communities)} communities")

# CREATE TIME SIMILARITY DATA 
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

#  NORMALIZE ITEM SIMILARITY MATRIX
def normalize_item_similarity_matrix(item_similarity_matrix):
    """
    Normalize the entire item similarity matrix to ensure pairwise similarities 
    have consistent scale.
    
    Returns:
        normalized_matrix: A normalized version of the similarity matrix with values in [0,1]
    """
    normalized_matrix = item_similarity_matrix.copy()

    min_sim = np.min(normalized_matrix)
    max_sim = np.max(normalized_matrix)

    if max_sim > min_sim:
        normalized_matrix = (normalized_matrix - min_sim) / (max_sim - min_sim)
    
    return normalized_matrix

# TIME-BASED COMMUNITY SIMILARITY FUNCTION 
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

#  MATRIX FACTORIZATION 
class MatrixFactorization:
    def __init__(self, num_users, num_items, num_factors=32, lr=0.01, reg=0.01, epochs=50, seed=None):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        
        np.random.seed(seed)
        self.user_factors = np.random.normal(0, 0.1, (num_users, num_factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, num_factors))

    def train(self, interaction_matrix):
        users, items = interaction_matrix.nonzero()

        for epoch in tqdm(range(self.epochs), desc="Training MF", leave=False):
            indices = np.arange(len(users))
            np.random.shuffle(indices)
            
            for idx in indices:
                u, i = users[idx], items[idx]
                pred = self.user_factors[u].dot(self.item_factors[i])
                err = 1 - pred  

                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * self.user_factors[u] - self.reg * self.item_factors[i])

    def predict(self, user_ids, item_ids):
        if isinstance(user_ids, (int, np.integer)):
            user_ids = [user_ids]
            item_ids = [item_ids]
            
        return np.sum(self.user_factors[user_ids] * self.item_factors[item_ids], axis=1)

# COMBINED RECOMMENDATIONS FUNCTION
def combined_recommendations(model, user_id, item_ids, item_similarity_matrix, 
                           user_item_timestamps, user_community, item_community,
                           normalized_item_sim_matrix, alpha_mf=0.7, alpha_item=0.2, alpha_time=0.1):
    """
    Apply community-based time similarity and item similarity to MF scores.
    
    Parameters:
    -----------
    alpha_mf : float
        Weight for MF model predictions
    alpha_item : float
        Weight for item similarity scores
    alpha_time : float
        Weight for time decay scores (community-only)
    """
    mf_scores = model.predict(
        np.repeat(user_id, len(item_ids)),
        np.array(item_ids)
    )
    
    item_sim_scores = np.zeros_like(mf_scores)
    for i, item_id in enumerate(item_ids):
        if item_id < normalized_item_sim_matrix.shape[0]:
            item_sim_scores[i] = np.mean(normalized_item_sim_matrix[item_id])
    
    time_scores = np.zeros_like(mf_scores)
    
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

    assert abs(alpha_mf + alpha_item + alpha_time - 1.0) < 1e-6, "Alpha weights must sum to 1.0"
    
    combined_scores = (
        alpha_mf * mf_scores + 
        alpha_item * item_sim_scores + 
        alpha_time * time_scores
    )
    
    return combined_scores

# EVALUATION FUNCTION 
def evaluate_mf_combined(model, test_interactions, train_interactions, item_similarity_matrix, 
                               user_item_timestamps, user_community, item_community,
                               normalized_item_sim_matrix, ks=[5, 10, 20], 
                               alpha_mf=0.7, alpha_item=0.2, alpha_time=0.1,
                               log_every_n=50):
    """
    Evaluates model with extensive logging to diagnose issues.
    """
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    
    for u, i in train_interactions:
        user_interactions[u].add(i)
    
    component_metrics = {
        'mf': {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks},
        'hybrid': {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    }
    
    users_evaluated = 0
    
    for user_idx, (u, true_i) in enumerate(test_interactions):
        log_this_user = (user_idx % log_every_n == 0)
        seen = user_interactions[u]
        candidates = list(set(range(n_items)) - seen)
        
        if len(candidates) < 99:
            sampled = candidates
        else:
            sampled = np.random.choice(candidates, 99, replace=False).tolist()
        
        test_items = sampled + [true_i]

        mf_scores = model.predict(
            np.repeat(u, len(test_items)),
            np.array(test_items)
        )

        hybrid_scores = combined_recommendations(
            model, u, test_items, item_similarity_matrix, 
            user_item_timestamps, user_community, item_community,
            normalized_item_sim_matrix, alpha_mf, alpha_item, alpha_time
        )
        
        mf_top_indices = np.argsort(mf_scores)[::-1]
        mf_top_items = [test_items[i] for i in mf_top_indices]
        
        hybrid_top_indices = np.argsort(hybrid_scores)[::-1]
        hybrid_top_items = [test_items[i] for i in hybrid_top_indices]

        for approach, top_items in [('mf', mf_top_items), ('hybrid', hybrid_top_items)]:
            for k in ks:
                actual_k = min(k, len(top_items))
                top_k = top_items[:actual_k]
                binary_relevance = [1 if i == true_i else 0 for i in top_k]
                
                # Calculate metrics
                hit = 1 if true_i in top_k else 0
                component_metrics[approach][k]['hit'] += hit
                
                if hit > 0:
                    rank = top_k.index(true_i) + 1
                    component_metrics[approach][k]['recall'] += 1.0 / rank
                    
                    ideal = sorted(binary_relevance, reverse=True)
                    try:
                        component_metrics[approach][k]['ndcg'] += ndcg_score(np.array([ideal]), np.array([binary_relevance]))
                    except Exception as e:
                        pass
        
        for k in ks:
            actual_k = min(k, len(hybrid_top_items))
            top_k = hybrid_top_items[:actual_k]
            binary_relevance = [1 if i == true_i else 0 for i in top_k]
            
            # Calculate metrics
            hit = 1 if true_i in top_k else 0
            metrics[k]['hit'] += hit
            
            if hit > 0:
                rank = top_k.index(true_i) + 1
                metrics[k]['recall'] += 1.0 / rank
                
                ideal = sorted(binary_relevance, reverse=True)
                try:
                    metrics[k]['ndcg'] += ndcg_score(np.array([ideal]), np.array([binary_relevance]))
                except Exception as e:
                    pass
        
        users_evaluated += 1

    results = {}
    for k in ks:
        results[k] = {
            'recall': metrics[k]['recall'] / users_evaluated,
            'ndcg': metrics[k]['ndcg'] / users_evaluated,
            'hit_rate': metrics[k]['hit'] / users_evaluated
        }
    
    return results

# MAIN EXECUTION 
if __name__ == "__main__":
    alpha_item_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    alpha_time_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    
    alpha_combinations = []
    for alpha_item in alpha_item_values:
        for alpha_time in alpha_time_values:
            alpha_mf = 1.0 - alpha_item - alpha_time
            if alpha_mf >= 0.0:  
                alpha_combinations.append((alpha_mf, alpha_item, alpha_time))
    
    alpha_combinations.sort(key=lambda x: x[0], reverse=True)
    
    num_runs = 3
    ks = [5, 10, 20]
    
    print(f"Starting grid search with {len(alpha_combinations)} alpha combinations, {num_runs} runs each")
    
    grid_search_results = []
    all_runs_results = []  

    normalized_item_sim_matrix = normalize_item_similarity_matrix(item_similarity_matrix)
    
    start_time = time.time()
    
    for combo_idx, (alpha_mf, alpha_item, alpha_time) in enumerate(tqdm(alpha_combinations, desc="Testing alpha combinations")):
        print(f"\nCombination {combo_idx+1}/{len(alpha_combinations)}: mf={alpha_mf:.3f}, item={alpha_item:.3f}, time={alpha_time:.3f}")

        aggregated_results = {k: {metric: 0.0 for metric in ['recall', 'ndcg', 'hit_rate']} for k in ks}
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}")
            
            np.random.seed(42 + run)
            
            model = MatrixFactorization(
                num_users=n_users,
                num_items=n_items,
                num_factors=32,
                lr=0.01,
                reg=0.1,
                epochs=100,
                seed=42 + run
            )
            model.train(train_matrix)
            
            try:
                run_results = evaluate_mf_combined(
                    model, 
                    test_interactions, 
                    train_interactions, 
                    item_similarity_matrix, 
                    user_item_timestamps, 
                    user_community, 
                    item_community,
                    normalized_item_sim_matrix,
                    ks=ks,
                    alpha_mf=alpha_mf,
                    alpha_item=alpha_item,
                    alpha_time=alpha_time
                )
                
                run_data = {
                    'alpha_mf': alpha_mf,
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

        result_row = {
            'alpha_mf': alpha_mf,
            'alpha_item': alpha_item,
            'alpha_time': alpha_time
        }
        
        for k in ks:
            for metric in ['recall', 'ndcg', 'hit_rate']:
                result_row[f"{metric}@{k}"] = aggregated_results[k][metric]
        
        grid_search_results.append(result_row)

        print(f"  Average results for mf={alpha_mf:.2f}, item={alpha_item:.2f}, time={alpha_time:.2f}:")
        for k in ks:
            print(f" @K={k} → " +
                  f"Recall: {aggregated_results[k]['recall']:.4f}, " +
                  f"NDCG: {aggregated_results[k]['ndcg']:.4f}, " +
                  f"Hit Rate: {aggregated_results[k]['hit_rate']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame(grid_search_results)
    csv_file = f"mf_community_time_grid_search_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    

    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_csv = f"mf_community_time_grid_search_all_runs_{timestamp}.csv"
    all_runs_df.to_csv(all_runs_csv, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Grid search complete in {elapsed_time/60:.2f} minutes.")
    print(f"Results saved to:")
    print(f"- Aggregated results: '{csv_file}'")
    print(f"- All individual runs: '{all_runs_csv}'")