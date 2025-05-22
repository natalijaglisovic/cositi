import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
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

# MAP USER & ITEM IDS 
user_map = {user: u for u, user in enumerate(df['userID'].unique())}
item_map = {item: i for i, item in enumerate(df['itemID'].unique())}

df["userID"] = df["userID"].map(user_map)
df["itemID"] = df["itemID"].map(item_map)
num_users = len(user_map)
num_items = len(item_map)

user_community = df.groupby('userID')['Community'].first().to_dict()
item_community = df.groupby('itemID')['Community'].first().to_dict()

print(f"Mapped {num_users} users and {num_items} items")

# TRAIN-TEST SPLIT
df["rank"] = df.groupby("userID").cumcount(ascending=False) + 1
train_df = df[df["rank"] > 1].drop(columns=["rank"])
test_df = df[df["rank"] == 1].drop(columns=["rank"])
print(f"Train set: {len(train_df)} interactions, Test set: {len(test_df)} interactions")

train_matrix = csr_matrix(
    (np.ones(len(train_df)), (train_df["userID"], train_df["itemID"])),
    shape=(num_users, num_items)
).toarray()

test_matrix = csr_matrix(
    (np.ones(len(test_df)), (test_df["userID"], test_df["itemID"])),
    shape=(num_users, num_items)
).toarray()

print(f"Created train matrix {train_matrix.shape} and test matrix {test_matrix.shape}")

train_interactions = list(zip(train_df['userID'], train_df['itemID']))
test_interactions = list(zip(test_df['userID'], test_df['itemID']))

user_item_timestamps = {}
for _, row in train_df.iterrows():
    user_id = row['userID']
    item_id = row['itemID']
    timestamp = row['event_date'].timestamp()
    
    if user_id not in user_item_timestamps:
        user_item_timestamps[user_id] = {}
    user_item_timestamps[user_id][item_id] = timestamp

# ITEM SIMILARITY 
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

# MATRIX FACTORIZATION MODEL 
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
        users, items = np.where(interaction_matrix > 0)
        
        for epoch in range(self.epochs):
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

# TIME SIMILARITY 
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

# COMMUNITY TIME RECOMMENDATIONS 
def community_time_recommendations(model, user_id, item_ids, user_item_timestamps, 
                                  user_community, item_community, alpha_time=0.5):
    """
    Apply community-based time similarity to MF scores for recommendation using a weighted approach.
    """
    alpha_mf = 1.0 - alpha_time
    
    mf_scores = model.predict(
        np.repeat(user_id, len(item_ids)),
        np.array(item_ids)
    )

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
    
    mf_range = np.max(mf_scores) - np.min(mf_scores)
    if mf_range > 1e-10:
        mf_scores = (mf_scores - np.min(mf_scores)) / mf_range
    
    time_range = np.max(time_scores) - np.min(time_scores)
    if time_range > 1e-10:
        time_scores = (time_scores - np.min(time_scores)) / time_range
    
    combined_scores = alpha_mf * mf_scores + alpha_time * time_scores
    
    return combined_scores

#  EVALUATIONS
def evaluate_with_similarity_and_alpha(model, train_matrix, test_matrix, 
                                       item_similarity_matrix, alpha_item=0.5,
                                       ks=[5, 10, 20], batch_size=100):
    """
    Evaluates the model using:
    - 99 random items + true test item
    - Combination of MF scores and item similarity with alpha parameter
    - Processes users in batches for efficiency
    """
    results = {k: {"ndcg": 0, "hit_rate": 0, "recall": 0, "users": 0} for k in ks}
    
    user_ids, item_ids = np.where(test_matrix > 0)
    unique_users = np.unique(user_ids)
    total_users = len(unique_users)
    
    all_items = set(range(test_matrix.shape[1]))
    
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
            
            mf_scores = model.predict(
                np.repeat(user_id, len(candidate_items)),
                np.array(candidate_items)
            )
            
            item_sim_scores = np.zeros_like(mf_scores)
            for idx, item_id in enumerate(candidate_items):
                similarities = []
                for past_item in user_train_items:
                    if past_item < item_similarity_matrix.shape[0] and item_id < item_similarity_matrix.shape[1]:
                        similarities.append(item_similarity_matrix[past_item, item_id])

                if similarities:
                    item_sim_scores[idx] = np.mean(similarities)
            
            mf_range = np.max(mf_scores) - np.min(mf_scores)
            if mf_range > 1e-10:
                mf_scores = (mf_scores - np.min(mf_scores)) / mf_range
            
            sim_range = np.max(item_sim_scores) - np.min(item_sim_scores)
            if sim_range > 1e-10:
                item_sim_scores = (item_sim_scores - np.min(item_sim_scores)) / sim_range

            combined_scores = (1 - alpha_item) * mf_scores + alpha_item * item_sim_scores
            
            for k in ks:
                actual_k = min(k, len(candidate_items))

                top_k_indices = np.argsort(-combined_scores)[:actual_k]
                top_k_items = [candidate_items[i] for i in top_k_indices]

                binary_relevance = [1 if i in test_items else 0 for i in top_k_items]
                ideal = sorted(binary_relevance, reverse=True)

                try:
                    ndcg_val = ndcg_score([ideal], [binary_relevance])
                    results[k]["ndcg"] += ndcg_val
                except Exception as e:
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

def evaluate_mf_with_community_time(model, test_interactions, train_interactions, 
                                   user_item_timestamps, user_community, item_community,
                                   ks=[5, 10, 20], alpha_time=0.5, batch_size=1000):
    """
    Evaluate MF model with community-based time similarity
    """
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    total_evaluated = 0

    for u, i in train_interactions:
        user_interactions[u].add(i)
    
    all_items = set(range(num_items))
    
    unique_test_users = list(set([u for u, _ in test_interactions]))
    num_batches = (len(unique_test_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(unique_test_users))
        batch_users = unique_test_users[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_users)} users)")
        
        for u in batch_users:
            test_items = [i for u_test, i in test_interactions if u_test == u]
            if not test_items:
                continue
                
            true_i = test_items[0]  
            
            if u not in user_interactions:
                continue
                
            seen = user_interactions[u]
            candidates = list(all_items - seen)

            if len(candidates) < 99:
                additional_candidates = list(seen - {true_i})
                candidates = (candidates + additional_candidates)[:99]
            else:
                candidates = np.random.choice(candidates, 99, replace=False).tolist()
            
            test_candidates = candidates + [true_i]
            
            
            scores = community_time_recommendations(
                model, u, test_candidates, user_item_timestamps, 
                user_community, item_community, alpha_time
            )

            top_all_indices = np.argsort(scores)[::-1]
            
            for k in ks:
                top_k_indices = top_all_indices[:k]
                top_k_items = [test_candidates[i] for i in top_k_indices]
                
                hit = 1 if true_i in top_k_items else 0
                metrics[k]['hit'] += hit

                if hit:
                    rank = top_k_items.index(true_i) + 1 
                    metrics[k]['recall'] += 1.0 / rank
                
                if hit:
                    binary_relevance = np.zeros(k)
                    binary_relevance[top_k_items.index(true_i)] = 1
                    
                    ideal_relevance = np.zeros(k)
                    ideal_relevance[0] = 1
                    
                    try:
                        ndcg = ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))
                        metrics[k]['ndcg'] += ndcg
                    except Exception as e:
                        pass
            
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
    alpha_item_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    alpha_time_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    num_runs = 3
    ks = [5, 10, 20]
    batch_size = 1000  
    
    # ITEM SIMILARITY 
    print("\n" + "="*50)
    print("RUNNING ABLATION 1: ITEM SIMILARITY")
    print("="*50)
    
    start_time = time.time()
    
    print(f"Starting MF with item similarity evaluation ({len(alpha_item_values)} alpha values, {num_runs} runs each)")
    
    item_sim_output_rows = []
    
    for alpha_item in tqdm(alpha_item_values, desc="Evaluating alpha_item values"):
        print(f"\nEvaluating alpha_item = {alpha_item}")
        
        run_results_all = []
        
        for run in range(1, num_runs + 1):
            print(f"Run {run}/{num_runs}")
            
            np.random.seed(42 + run)
            model = MatrixFactorization(
                num_users=num_users,
                num_items=num_items,
                num_factors=32,
                lr=0.01,
                reg=0.1,
                epochs=100,
                seed=42 + run
            )
            model.train(train_matrix)

            run_metrics = evaluate_with_similarity_and_alpha(
                model, 
                train_matrix,
                test_matrix,
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
        
        item_sim_output_rows.append(row)
        
        print(f"  Average results for alpha_item = {alpha_item}:")
        for k in ks:
            print(f" @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    df_item_sim = pd.DataFrame(item_sim_output_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    item_sim_csv = f"mf_alpha_item_similarity_amazon_health_household.csv"
    df_item_sim.to_csv(item_sim_csv, index=False)
    
    ablation1_elapsed = time.time() - start_time
    print(f"Ablation 1 complete in {ablation1_elapsed/60:.2f} minutes. Results saved to '{item_sim_csv}'")
    print("\nAblation 1 Summary:")
    print(df_item_sim.to_string())
    
    
    # TIME-BASED COMMUNITY SIMILARITY
    print("\n" + "="*50)
    print("RUNNING ABLATION 2: TIME-BASED COMMUNITY SIMILARITY")
    print("="*50)
    
    start_time = time.time()
    
    print(f"Starting MF with time-based community evaluation ({len(alpha_time_values)} alpha values, {num_runs} runs each)")
    
    np.random.seed(42)
    
    time_sim_output_rows = []
    
    for alpha_time in tqdm(alpha_time_values, desc="Evaluating alpha_time values"):
        print(f"\nEvaluating alpha_time = {alpha_time}")
        
        run_results_all = []
        
        for run in range(1, num_runs + 1):
            print(f"Run {run}/{num_runs}")
            
            np.random.seed(42 + run)
            model = MatrixFactorization(
                num_users=num_users,
                num_items=num_items,
                num_factors=32,
                lr=0.01,
                reg=0.1,
                epochs=100,
                seed=42 + run
            )
            model.train(train_matrix)
            
            run_metrics = evaluate_mf_with_community_time(
                model, 
                test_interactions,
                train_interactions,
                user_item_timestamps, 
                user_community, 
                item_community,
                ks=ks,
                alpha_time=alpha_time,
                batch_size=batch_size
            )
            
            run_results_all.append(run_metrics)

            info = f"  Results: "
            for k in ks:
                info += f"R@{k}={run_metrics[k]['recall']:.4f}, "
                info += f"NDCG@{k}={run_metrics[k]['ndcg']:.4f}, "
                info += f"HR@{k}={run_metrics[k]['hit_rate']:.4f} | "
            print(info)
        
        row = {"alpha_time": alpha_time}
        
        for k in ks:
            recall_avg = np.mean([results[k]['recall'] for results in run_results_all])
            ndcg_avg = np.mean([results[k]['ndcg'] for results in run_results_all])
            hit_rate_avg = np.mean([results[k]['hit_rate'] for results in run_results_all])
            row[f"recall@{k}"] = recall_avg
            row[f"ndcg@{k}"] = ndcg_avg
            row[f"hit_rate@{k}"] = hit_rate_avg
        
        time_sim_output_rows.append(row)

        print(f"  Average results for alpha_item = {alpha_time}:")
        for k in ks:
            print(f" @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    df_time = pd.DataFrame(time_sim_output_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_sim_csv = f"mf_alpha_time_similarity_amazon_health_household.csv"
    df_time.to_csv(time_sim_csv, index=False)
    
    