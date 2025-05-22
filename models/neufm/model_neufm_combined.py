import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import time
import itertools
from datetime import datetime

#  LOAD DATA 
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

# COMPUTE ITEM SIMILARITY MATRIX
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

#  CREATE TIME SIMILARITY DATA 
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

# NEUMF DATASET
class NeuMFDataset(Dataset):
    def __init__(self, interactions, n_items, num_negatives=4):
        self.data = []
        user_pos = defaultdict(set)
        for u, i in interactions:
            user_pos[u].add(i)
        for u, i in interactions:
            self.data.append((u, i, 1))
            for _ in range(num_negatives):
                j = np.random.randint(n_items)
                while j in user_pos[u]:
                    j = np.random.randint(n_items)
                self.data.append((u, j, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, label = self.data[idx]
        return torch.LongTensor([u]), torch.LongTensor([i]), torch.FloatTensor([label])

#  DEFINE NEUMF MODEL
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super(NeuMF, self).__init__()
        
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim)
    
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)

        nn.init.xavier_uniform_(self.gmf_user_emb.weight)
        nn.init.xavier_uniform_(self.gmf_item_emb.weight)
        nn.init.xavier_uniform_(self.mlp_user_emb.weight)
        nn.init.xavier_uniform_(self.mlp_item_emb.weight)

        self.fc1 = nn.Linear(emb_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

        self.neuMF = nn.Linear(emb_dim + 16, 1)
    
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.neuMF.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.neuMF.bias)

    def forward(self, users, items):
        gmf_user_emb = self.gmf_user_emb(users)
        gmf_item_emb = self.gmf_item_emb(items)
        mlp_user_emb = self.mlp_user_emb(users)
        mlp_item_emb = self.mlp_item_emb(items)

        if gmf_user_emb.dim() == 3:
            gmf_user_emb = gmf_user_emb.squeeze(1)
            gmf_item_emb = gmf_item_emb.squeeze(1)
            mlp_user_emb = mlp_user_emb.squeeze(1)
            mlp_item_emb = mlp_item_emb.squeeze(1)

        gmf_vector = gmf_user_emb * gmf_item_emb
 
        mlp_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_vector = F.relu(self.fc1(mlp_vector))
        mlp_vector = F.relu(self.fc2(mlp_vector))
        mlp_vector = F.relu(self.fc3(mlp_vector))
  
        neuMF_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        pred = torch.sigmoid(self.neuMF(neuMF_vector))
        
        return pred.squeeze()

# TRAINING FUNCTION 
def train_model(model, dataloader, epochs=15, lr=0.01, verbose=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for users, items, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(users, items)
            
            try:
                loss = loss_fn(outputs, labels.squeeze())
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                else:
                    if verbose:
                        print(f"Warning: Invalid loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
            except Exception as e:
                if verbose:
                    print(f"Error in training batch: {e}")
            
            batch_count += 1
        if batch_count > 0 and verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/batch_count:.4f}")
        elif batch_count == 0 and verbose:
            print(f"Epoch {epoch+1}/{epochs}, No valid batches processed")

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

# NORMALIZED ITEM SIMILARITY MATRIX
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

# COMBINED RECOMMENDATIONS FUNCTION 
def combined_recommendations(model, user_id, item_ids, item_similarity_matrix, 
                           user_item_timestamps, user_community, item_community,
                           normalized_item_sim_matrix, alpha_neumf=0.7, alpha_item=0.2, alpha_time=0.1):
    """
    Apply community-based time similarity and item similarity to NeuMF scores.
    
    Parameters:
    -----------
    alpha_neumf : float
        Weight for NeuMF model predictions
    alpha_item : float
        Weight for item similarity scores
    alpha_time : float
        Weight for time decay scores (community-only)
    """
    model.eval()
    user_tensor = torch.LongTensor([user_id] * len(item_ids))
    item_tensor = torch.LongTensor(item_ids)
    
    with torch.no_grad():
        neumf_scores = model(user_tensor, item_tensor).detach().cpu().numpy()
    
    item_sim_scores = np.zeros_like(neumf_scores)
    for i, item_id in enumerate(item_ids):
        if item_id < normalized_item_sim_matrix.shape[0]:
            item_sim_scores[i] = np.mean(normalized_item_sim_matrix[item_id])
    
    time_scores = np.zeros_like(neumf_scores)
    
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
    
    assert abs(alpha_neumf + alpha_item + alpha_time - 1.0) < 1e-6, "Alpha weights must sum to 1.0"
    
    combined_scores = (
        alpha_neumf * neumf_scores + 
        alpha_item * item_sim_scores + 
        alpha_time * time_scores
    )
    
    return combined_scores

#  EVALUATION 
def evaluate_neumf_combined(model, test_interactions, train_set, item_similarity_matrix, 
                                  user_item_timestamps, user_community, item_community,
                                  normalized_item_sim_matrix, ks=[5, 10, 20], 
                                  alpha_neumf=0.985, alpha_item=0.01, alpha_time=0.005,
                                  log_every_n=50):
    """
    Evaluates model with extensive logging to diagnose issues.
    """
    model.eval()
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    for u, i in train_set:
        user_interactions[u].add(i)

    component_metrics = {
        'neumf': {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks},
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
        
        user_tensor = torch.LongTensor([u] * len(test_items))
        item_tensor = torch.LongTensor(test_items)
        with torch.no_grad():
            neumf_scores = model(user_tensor, item_tensor).detach().cpu().numpy()
        
        hybrid_scores = combined_recommendations(
            model, u, test_items, item_similarity_matrix, 
            user_item_timestamps, user_community, item_community,
            normalized_item_sim_matrix, alpha_neumf, alpha_item, alpha_time
        )
        
        neumf_top_indices = np.argsort(neumf_scores)[::-1]
        neumf_top_items = [test_items[i] for i in neumf_top_indices]
        
        hybrid_top_indices = np.argsort(hybrid_scores)[::-1]
        hybrid_top_items = [test_items[i] for i in hybrid_top_indices]
        
        for approach, top_items in [('neumf', neumf_top_items), ('hybrid', hybrid_top_items)]:
            for k in ks:
                actual_k = min(k, len(top_items))
                top_k = top_items[:actual_k]
                binary_relevance = [1 if i == true_i else 0 for i in top_k]
                
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

#  MAIN EXECUTION
if __name__ == "__main__":
    alpha_item_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    alpha_time_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]

    alpha_combinations = []
    for alpha_item in alpha_item_values:
        for alpha_time in alpha_time_values:
            alpha_neumf = 1.0 - alpha_item - alpha_time
            if alpha_neumf >= 0.0: 
                alpha_combinations.append((alpha_neumf, alpha_item, alpha_time))
    
    alpha_combinations.sort(key=lambda x: x[0], reverse=True)
    num_runs = 3
    ks = [5, 10, 20]
    
    print(f"Starting grid search with {len(alpha_combinations)} alpha combinations, {num_runs} runs each")

    grid_search_results = []
    all_runs_results = []  

    normalized_item_sim_matrix = normalize_item_similarity_matrix(item_similarity_matrix)
    
    start_time = time.time()
    
    for combo_idx, (alpha_neumf, alpha_item, alpha_time) in enumerate(tqdm(alpha_combinations, desc="Testing alpha combinations")):
        print(f"\nCombination {combo_idx+1}/{len(alpha_combinations)}: neumf={alpha_neumf:.3f}, item={alpha_item:.3f}, time={alpha_time:.3f}")
        aggregated_results = {k: {metric: 0.0 for metric in ['recall', 'ndcg', 'hit_rate']} for k in ks}
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}")
        
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            train_dataset = NeuMFDataset(train_interactions, n_items)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            
            model = NeuMF(n_users=n_users, n_items=n_items, emb_dim=64)
            train_model(model, train_loader, epochs=15, verbose=False)

            try:
                run_results = evaluate_neumf_combined(
                    model, 
                    test_interactions, 
                    train_interactions, 
                    item_similarity_matrix, 
                    user_item_timestamps, 
                    user_community, 
                    item_community,
                    normalized_item_sim_matrix,
                    ks=ks,
                    alpha_neumf=alpha_neumf,
                    alpha_item=alpha_item,
                    alpha_time=alpha_time
                )
                run_data = {
                    'alpha_neumf': alpha_neumf,
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
            'alpha_neumf': alpha_neumf,
            'alpha_item': alpha_item,
            'alpha_time': alpha_time
        }
        
        for k in ks:
            for metric in ['recall', 'ndcg', 'hit_rate']:
                result_row[f"{metric}@{k}"] = aggregated_results[k][metric]
        
        grid_search_results.append(result_row)
        print(f"  Average results for neumf={alpha_neumf:.2f}, item={alpha_item:.2f}, time={alpha_time:.2f}:")
        for k in ks:
            print(f"@K={k} → " +
                  f"Recall: {aggregated_results[k]['recall']:.4f}, " +
                  f"NDCG: {aggregated_results[k]['ndcg']:.4f}, " +
                  f"Hit Rate: {aggregated_results[k]['hit_rate']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame(grid_search_results)
    csv_file = f"neumf_community_time_grid_search_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
 
    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_csv = f"neumf_community_time_grid_search_all_runs_{timestamp}.csv"
    all_runs_df.to_csv(all_runs_csv, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Grid search complete in {elapsed_time/60:.2f} minutes.")
    print(f"Results saved to:")
    print(f"- Aggregated results: '{csv_file}'")
    print(f"- All individual runs: '{all_runs_csv}'")