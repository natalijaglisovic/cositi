import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

# LOAD & PREPROCESS DATA 
file_path = "data/amazon_health_household_communities.csv"  # Adjust path as needed
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
dataset.fit(user_map.values(), item_map.values())

(interactions_train, _) = dataset.build_interactions([
    (row.userID, row.itemID) for _, row in train_df.iterrows()
])
(interactions_test, _) = dataset.build_interactions([
    (row.userID, row.itemID) for _, row in test_df.iterrows()
])

train_matrix = interactions_train.toarray()
test_matrix = interactions_test.toarray()

# PREPARE ITEM SIMILARITY

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

category_dummies = pd.get_dummies(item_features_df['category'], prefix='category')
brand_dummies = pd.get_dummies(item_features_df['brand'], prefix='brand')
community_dummies = pd.get_dummies(item_features_df['Community'], prefix='Community')

item_features = pd.concat([category_dummies, brand_dummies, community_dummies], axis=1)

item_similarity_matrix = cosine_similarity(item_features.to_numpy())
print(f"Created item similarity matrix of shape {item_similarity_matrix.shape}")

item_to_community = dict(zip(item_features_df['itemID'], item_features_df['Community']))

communities = item_features_df['Community'].unique()
items_by_community = {}
for comm in communities:
    items_by_community[comm] = item_features_df[item_features_df['Community'] == comm]['itemID'].values

print(f"Prepared community mappings for {len(communities)} communities")

# PREPARE TIME SIMILARITY DATA 
user_item_timestamps = {}
for _, row in train_df.iterrows():
    user_id = row['userID']
    item_id = row['itemID']
    timestamp = row['event_date'].timestamp()
    
    if user_id not in user_item_timestamps:
        user_item_timestamps[user_id] = {}
    user_item_timestamps[user_id][item_id] = timestamp

# TIME SIMILARITY FUNCTION
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

# EVALUATION FUNCTIONS

def evaluate_with_item_similarity(model, interactions_train, interactions_test, 
                               item_similarity_matrix, alpha_item=0.5,
                               ks=[5, 10, 20], batch_size=1000):
    """
    Evaluates the model using:
    - 99 random items + true test item
    - Combination of BPR scores and item similarity with alpha_item parameter
    - Processes users in batches for efficiency
    """
    results = {k: {"ndcg": 0, "hit_rate": 0, "recall": 0, "users": 0} for k in ks}
    
    user_ids, item_ids = interactions_test.nonzero()
    unique_users = np.unique(user_ids)
    total_users = len(unique_users)
    
    num_items = test_matrix.shape[1]
    all_items = set(range(num_items))
    
    num_batches = (total_users + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_users)
        batch_users = unique_users[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_users)} users)")
        
        for user_id in batch_users:
            test_items = np.where(test_matrix[user_id] > 0)[0]
            if len(test_items) == 0:
                continue

            user_train_items = np.where(train_matrix[user_id] > 0)[0]
            seen_items = set(user_train_items)

            candidates = list(all_items - seen_items)

            if len(candidates) < 99:
                additional_candidates = list(seen_items - set(test_items))
                candidates = (candidates + additional_candidates)[:99]
            else:
                candidates = np.random.choice(candidates, 99, replace=False).tolist()
            
            candidate_items = candidates + list(test_items)
            
            bpr_scores = model.predict(
                np.repeat(user_id, len(candidate_items)),
                np.array(candidate_items)
            )
            
            item_sim_scores = np.zeros_like(bpr_scores)
            for idx, item_id in enumerate(candidate_items):
                similarities = []
                for past_item in user_train_items:
                    if past_item < item_similarity_matrix.shape[0] and item_id < item_similarity_matrix.shape[1]:
                        similarities.append(item_similarity_matrix[past_item, item_id])
                
                if similarities:
                    item_sim_scores[idx] = np.mean(similarities)
            
            bpr_range = np.max(bpr_scores) - np.min(bpr_scores)
            if bpr_range > 1e-10:
                bpr_scores = (bpr_scores - np.min(bpr_scores)) / bpr_range
            
            sim_range = np.max(item_sim_scores) - np.min(item_sim_scores)
            if sim_range > 1e-10:
                item_sim_scores = (item_sim_scores - np.min(item_sim_scores)) / sim_range

            combined_scores = (1 - alpha_item) * bpr_scores + alpha_item * item_sim_scores

            for k in ks:
                actual_k = min(k, len(candidate_items))
                
                top_k_indices = np.argsort(-combined_scores)[:actual_k]
                top_k_items = [candidate_items[i] for i in top_k_indices]

                binary_relevance = [1 if i in test_items else 0 for i in top_k_items]
                ideal = sorted(binary_relevance, reverse=True)
                
                try:
                    ndcg_val = ndcg_score([ideal], [binary_relevance])
                    results[k]["ndcg"] += ndcg_val
                except:
                    pass

                hits = np.intersect1d(test_items, top_k_items)
                hit = len(hits) > 0
                results[k]["hit_rate"] += int(hit)
                
                if hit:
                    for idx, item_id in enumerate(top_k_items):
                        if item_id in test_items:
                            rank = idx + 1
                            break
                    modified_recall = 1.0 / rank
                    results[k]["recall"] += modified_recall
                
                results[k]["users"] += 1
    
    for k in ks:
        if results[k]["users"] > 0:
            results[k]["ndcg"] /= results[k]["users"]
            results[k]["hit_rate"] /= results[k]["users"]
            results[k]["recall"] /= results[k]["users"]
        else:
            results[k]["ndcg"] = 0
            results[k]["hit_rate"] = 0
            results[k]["recall"] = 0
    
    return {k: (results[k]["recall"], results[k]["ndcg"], results[k]["hit_rate"]) for k in ks}


def evaluate_with_time_similarity(model, interactions_train, interactions_test, 
                                  user_item_timestamps, user_community, item_community, 
                                  ks=[5, 10, 20], alpha_time=0.5, batch_size=1000):
    """
    Time-based evaluation function to be consistent with the item similarity implementation.
    Uses batched processing and consistent evaluation methodology.
    """
    results = {k: {"ndcg": 0, "hit_rate": 0, "recall": 0, "users": 0} for k in ks}
    
    user_ids, item_ids = interactions_test.nonzero()
    unique_users = np.unique(user_ids)
    total_users = len(unique_users)

    user_train_items = {}
    train_user_ids, train_item_ids = interactions_train.nonzero()
    for u, i in zip(train_user_ids, train_item_ids):
        if u not in user_train_items:
            user_train_items[u] = []
        user_train_items[u].append(i)
    
    n_users, n_items = interactions_train.shape
    all_items = set(range(n_items))
    
    num_batches = (total_users + batch_size - 1) // batch_size
    
    skipped_users = 0
    processed_users = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_users)
        batch_users = unique_users[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_users)} users)")
        
        for user_id in batch_users:
            test_items = np.where(test_matrix[user_id] > 0)[0]
            if len(test_items) == 0:
                skipped_users += 1
                continue
                
            if user_id not in user_train_items or not user_train_items[user_id]:
                skipped_users += 1
                continue
                
            user_items = user_train_items[user_id]
            seen_items = set(user_items)
            
            candidates = list(all_items - seen_items)
            
            if len(candidates) < 99:
                additional_candidates = list(seen_items - set(test_items))
                candidates = (candidates + additional_candidates)[:99]
            else:
                candidates = np.random.choice(candidates, 99, replace=False).tolist()

            candidate_items = candidates + list(test_items)

            bpr_scores = model.predict(
                np.repeat(user_id, len(candidate_items)),
                np.array(candidate_items)
            )
            
            time_scores = np.zeros_like(bpr_scores)

            if user_id in user_item_timestamps:
                for idx, item_id in enumerate(candidate_items):

                    similarities = []
                    for past_item in user_items:
                        if item_community.get(past_item) == item_community.get(item_id):
                            sim = calculate_time_similarity(
                                user_id, past_item, item_id,
                                user_item_timestamps, user_community, item_community
                            )
                            similarities.append(sim)

                    if similarities:
                        time_scores[idx] = np.mean(similarities)

            bpr_range = np.max(bpr_scores) - np.min(bpr_scores)
            if bpr_range > 1e-10:
                bpr_scores = (bpr_scores - np.min(bpr_scores)) / bpr_range
            
            time_range = np.max(time_scores) - np.min(time_scores)
            if time_range > 1e-10:
                time_scores = (time_scores - np.min(time_scores)) / time_range

            combined_scores = (1 - alpha_time) * bpr_scores + alpha_time * time_scores
            
            for k in ks:
                actual_k = min(k, len(candidate_items))
                
                top_k_indices = np.argsort(-combined_scores)[:actual_k]
                top_k_items = [candidate_items[i] for i in top_k_indices]
                
                binary_relevance = [1 if i in test_items else 0 for i in top_k_items]
                ideal = sorted(binary_relevance, reverse=True)
                
                try:
                    ndcg_val = ndcg_score([ideal], [binary_relevance])
                    results[k]["ndcg"] += ndcg_val
                except:
                    pass
                
                hits = np.intersect1d(test_items, top_k_items)
                hit = len(hits) > 0
                results[k]["hit_rate"] += int(hit)
                
                if hit:
                    for idx, item_id in enumerate(top_k_items):
                        if item_id in test_items:
                            rank = idx + 1
                            break
                    results[k]["recall"] += 1.0 / rank
                
                results[k]["users"] += 1
            
            processed_users += 1
    
    for k in ks:
        users_evaluated = results[k]["users"]
        if users_evaluated > 0:
            results[k]["ndcg"] /= users_evaluated
            results[k]["hit_rate"] /= users_evaluated
            results[k]["recall"] /= users_evaluated
        else:
            results[k]["ndcg"] = 0
            results[k]["hit_rate"] = 0
            results[k]["recall"] = 0
    
    return {k: (results[k]["recall"], results[k]["ndcg"], results[k]["hit_rate"]) for k in ks}

# MAIN EXECUTION 
if __name__ == "__main__":
    num_runs = 3
    ks = [5, 10, 20]  
    alpha_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    batch_size = 1000 
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ITEM SIMILARITY EVALUATION
    print("\n" + "="*80)
    print("RUNNING ITEM-BASED SIMILARITY EVALUATION")
    print("="*80)
    
    models = []
    for run in range(1, num_runs + 1):
        print(f"Training model for run {run}/{num_runs}")
        np.random.seed(42 + run)
        model = LightFM(loss="logistic", no_components=64, learning_rate=0.01, random_state=42 + run)
        model.fit(interactions_train, epochs=100, num_threads=8, verbose=False)
        models.append(model)
    
    print(f"Starting evaluation with {len(alpha_values)} alpha values, {num_runs} runs each")
    
    item_output_rows = []
    
    for alpha_item in tqdm(alpha_values, desc="Evaluating alpha_item values"):
        print(f"\nEvaluating alpha_item = {alpha_item}")
        
        run_results_all = []
        
        for run, model in enumerate(models, 1):
            print(f"Run {run}/{num_runs}")
            
            run_metrics = evaluate_with_item_similarity(
                model, 
                interactions_train,
                interactions_test,
                item_similarity_matrix,
                alpha_item=alpha_item, 
                ks=ks,
                batch_size=batch_size
            )
            
            run_results_all.append(run_metrics)
            
            info = f"  Results: "
            for k in ks:
                recall, ndcg, hit_rate = run_metrics[k]
                info += f"R@{k}={recall:.4f}, "
                info += f"NDCG@{k}={ndcg:.4f}, "
                info += f"HR@{k}={hit_rate:.4f} | "
            print(info)
        
        row = {"alpha_item": alpha_item}  
        
        for k in ks:
            recall_avg = np.mean([results[k][0] for results in run_results_all])
            ndcg_avg = np.mean([results[k][1] for results in run_results_all])
            hit_rate_avg = np.mean([results[k][2] for results in run_results_all])
            
            row[f"recall@{k}"] = recall_avg
            row[f"ndcg@{k}"] = ndcg_avg
            row[f"hit_rate@{k}"] = hit_rate_avg
        
        item_output_rows.append(row)

        print(f"  Average results for alpha_item = {alpha_item}:")  
        for k in ks:
            print(f"    @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    item_df = pd.DataFrame(item_output_rows)
    item_csv_file = f"bpr_item_similarity_amazon_health_household.csv"
    item_df.to_csv(item_csv_file, index=False)
    print(f"\nItem similarity results saved to '{item_csv_file}'")
    
    #  TIME SIMILARITY EVALUATION
    print("\n" + "="*80)
    print("RUNNING TIME-BASED SIMILARITY EVALUATION")
    print("="*80)
    
    time_output_rows = []
    
    for alpha_time in tqdm(alpha_values, desc="Evaluating alpha_time values"):
        print(f"\nEvaluating alpha_time = {alpha_time}")
        
        run_results_all = []
        
        for run, model in enumerate(models, 1):
            print(f"Run {run}/{num_runs}")

            run_metrics = evaluate_with_time_similarity(
                model, 
                interactions_train,
                interactions_test, 
                user_item_timestamps, 
                user_community,
                item_community,
                ks,
                alpha_time=alpha_time,
                batch_size=batch_size
            )
            
            run_results_all.append(run_metrics)

            info = f"  Results: "
            for k in ks:
                recall, ndcg, hit_rate = run_metrics[k]
                info += f"R@{k}={recall:.4f}, "
                info += f"NDCG@{k}={ndcg:.4f}, "
                info += f"HR@{k}={hit_rate:.4f} | "
            print(info)
        
        row = {"alpha_time": alpha_time}
        
        for k in ks:
            recall_avg = np.mean([results[k][0] for results in run_results_all])
            ndcg_avg = np.mean([results[k][1] for results in run_results_all])
            hit_rate_avg = np.mean([results[k][2] for results in run_results_all])
            
            row[f"recall@{k}"] = recall_avg
            row[f"ndcg@{k}"] = ndcg_avg
            row[f"hit_rate@{k}"] = hit_rate_avg
        
        time_output_rows.append(row)
        
        print(f"  Average results for alpha_time = {alpha_time}:")
        for k in ks:
            print(f"    @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    time_df = pd.DataFrame(time_output_rows)
    time_csv_file = f"bpr_time_similarity_amazon_health_household.csv"
    time_df.to_csv(time_csv_file, index=False)
    print(f"\nTime similarity results saved to '{time_csv_file}'")