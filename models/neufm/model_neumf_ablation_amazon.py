import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import datetime

# LOAD & PREPROCESS DATA 
file_path = "data/amazon_health_household_communities.csv"
usecols = ["userID", "itemID", "timestamp", "Community", 'brand', 'category']
df = pd.read_csv(file_path, sep=",", usecols=usecols, low_memory=False)

df['event_date'] = pd.to_datetime(df['timestamp'])
most_recent_date = df['event_date'].max()
df['days_since'] = (most_recent_date - df['event_date']).dt.days

print(f"Data loaded: {len(df)} interactions, {df['userID'].nunique()} users, {df['itemID'].nunique()} items")


#  MAP USER & ITEM IDs
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

#  DATASET 
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

# NeuMF MODEL 
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

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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
        gmf_sum = gmf_vector.sum(dim=1)  

        mlp_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_vector).squeeze()

        combined_out = gmf_sum + mlp_output
        pred = torch.sigmoid(combined_out)
        
        return pred

# TRAINING FUNCTION
def train_model(model, dataloader, epochs=15, lr=0.01, verbose=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for users, items, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(users, items)
            loss = loss_fn(outputs, labels.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        if verbose and batch_count > 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/batch_count:.4f}")
        elif epoch == epochs - 1 and not verbose:  
            print(f"Training complete: Final Loss: {total_loss/batch_count:.4f}")

# ITEM SIMILARITY 
def calculate_item_similarity_matrix(df):
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
    
    return item_similarity_matrix, item_to_community, items_by_community, communities

# CREATE TIME-BASED USER-ITEM TIMESTAMPS 
def create_time_user_item_data(train_df):
    user_community = train_df.groupby('userID')['Community'].first().to_dict()
    item_community = train_df.groupby('itemID')['Community'].first().to_dict()

    user_item_timestamps = {}
    for _, row in train_df.iterrows():
        user_id = row['userID']
        item_id = row['itemID']
        timestamp = row['event_date'].timestamp()
        
        if user_id not in user_item_timestamps:
            user_item_timestamps[user_id] = {}
        user_item_timestamps[user_id][item_id] = timestamp
    
    return user_item_timestamps, user_community, item_community

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

# = EVALUATION WITH ITEM SIMILARITY 
def evaluate_neumf_with_similarity(model, test_interactions, train_set, item_similarity_matrix, ks=[5, 10, 20], alpha_item=0.5):
    model.eval()
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    for u, i in train_set:
        user_interactions[u].add(i)

    with torch.no_grad():
        for u, true_i in test_interactions:
            seen = user_interactions[u]
            candidates = list(set(range(n_items)) - seen)
            if len(candidates) < 99:
                sampled = candidates
            else:
                sampled = np.random.choice(candidates, 99, replace=False).tolist()
                
            test_items = sampled + [true_i]

            user_tensor = torch.LongTensor([u] * len(test_items))
            item_tensor = torch.LongTensor(test_items)

            neuf_scores = model(user_tensor, item_tensor).detach().cpu().numpy()

            item_sim_scores = np.zeros_like(neuf_scores)
            for idx, item_id in enumerate(test_items):
                similarities = []
                for past_item in user_interactions[u]:
                    if past_item < item_similarity_matrix.shape[0] and item_id < item_similarity_matrix.shape[1]:
                        similarities.append(item_similarity_matrix[past_item, item_id])

                if similarities:
                    item_sim_scores[idx] = np.mean(similarities)
     
            neuf_range = np.max(neuf_scores) - np.min(neuf_scores)
            if neuf_range > 1e-10:
                neuf_scores = (neuf_scores - np.min(neuf_scores)) / neuf_range
            
            sim_range = np.max(item_sim_scores) - np.min(item_sim_scores)
            if sim_range > 1e-10:
                item_sim_scores = (item_sim_scores - np.min(item_sim_scores)) / sim_range
            fused_scores = (1 - alpha_item) * neuf_scores + alpha_item * item_sim_scores

            top_k_indices = np.argsort(fused_scores)[::-1]
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
                        print(f"NDCG error: {e}")
                        pass

    users_evaluated = len(test_interactions)
    results = {}
    for k in ks:
        results[k] = {
            'recall': metrics[k]['recall'] / users_evaluated,
            'ndcg': metrics[k]['ndcg'] / users_evaluated,
            'hit_rate': metrics[k]['hit'] / users_evaluated
        }
    
    return results

# COMMUNITY-BASED TIME RECOMMENDATIONS FUNCTION
def community_time_recommendations(model, user_id, item_ids, user_item_timestamps, 
                                  user_community, item_community, alpha_time=0.5):
    """
    Apply community-based time similarity to NeuMF scores for recommendation using a weighted average approach.
    """
    alpha_neuf = 1.0 - alpha_time
    
    model.eval()
    user_tensor = torch.LongTensor([user_id] * len(item_ids))
    item_tensor = torch.LongTensor(item_ids)
    
    with torch.no_grad():
        neuf_scores = model(user_tensor, item_tensor).detach().cpu().numpy()

    time_scores = np.zeros_like(neuf_scores)

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
    
    neuf_range = np.max(neuf_scores) - np.min(neuf_scores)
    if neuf_range > 1e-10:
        neuf_scores = (neuf_scores - np.min(neuf_scores)) / neuf_range
    
    time_range = np.max(time_scores) - np.min(time_scores)
    if time_range > 1e-10:
        time_scores = (time_scores - np.min(time_scores)) / time_range

    fused_scores = alpha_neuf * neuf_scores + alpha_time * time_scores
    
    return fused_scores

#  EVALUATION WITH COMMUNITY TIME
def evaluate_neumf_with_community_time(model, test_interactions, train_interactions, 
                                      user_item_timestamps, user_community, item_community,
                                      ks=[5, 10, 20], alpha_time=0.5):
    """
    Evaluate NeuMF model with community-based time similarity
    """
    model.eval()
    metrics = {k: {'hit': 0, 'ndcg': 0.0, 'recall': 0.0} for k in ks}
    user_interactions = defaultdict(set)
    for u, i in train_interactions:
        user_interactions[u].add(i)
    
    all_items = set(range(n_items))
    
    with torch.no_grad():
        for user_idx, (u, true_i) in enumerate(test_interactions):
            seen = user_interactions[u]
            candidates = list(all_items - seen)
            if len(candidates) < 99:
                additional_candidates = list(seen - {true_i})
                candidates = (candidates + additional_candidates)[:99]
            else:
                candidates = np.random.choice(candidates, 99, replace=False).tolist()
            test_items = candidates + [true_i]
            
            scores = community_time_recommendations(
                model, u, test_items, user_item_timestamps, 
                user_community, item_community, alpha_time
            )
            
            top_k_indices = np.argsort(scores)[::-1]
            top_k_items = [test_items[i] for i in top_k_indices]

            for k in ks:
                actual_k = min(k, len(top_k_items))
                top_k = top_k_items[:actual_k]
                binary_relevance = [1 if i == true_i else 0 for i in top_k]
                ideal = sorted(binary_relevance, reverse=True)
                hit = 1 if true_i in top_k else 0
                metrics[k]['hit'] += hit

                if true_i in top_k:
                    rank = top_k.index(true_i) + 1
                    modified_recall = 1.0 / rank  
                    metrics[k]['recall'] += modified_recall

                if hit > 0:
                    try:
                        metrics[k]['ndcg'] += ndcg_score([ideal], [binary_relevance])
                    except:
                        pass

    users_evaluated = len(test_interactions)
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
    alpha_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    num_runs = 3
    ks = [5, 10, 20]  
    
    # ITEM SIMILARITY 
    print("\n" + "="*40)
    print("ABLATION 1: NEUMF WITH ITEM SIMILARITY")
    print("="*40 + "\n")
    
    item_similarity_matrix, item_to_community, items_by_community, communities = calculate_item_similarity_matrix(df)
    item_sim_output_rows = []
    
    start_time = time.time()
    
    for alpha_item in tqdm(alpha_values, desc="Evaluating alpha_item values"):
        aggregated_results = {k: {metric: 0.0 for metric in ['recall', 'ndcg', 'hit_rate']} for k in ks}
        total_runs = 0
        
        print(f"\nEvaluating alpha_item = {alpha_item}")
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}")
            
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            train_dataset = NeuMFDataset(train_interactions, n_items)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            
            model = NeuMF(n_users=n_users, n_items=n_items, emb_dim=64)
            train_model(model, train_loader, epochs=15, verbose=False)
            
            run_results = evaluate_neumf_with_similarity(
                model, 
                test_interactions, 
                train_interactions, 
                item_similarity_matrix, 
                ks=ks,
                alpha_item=alpha_item
            )
            
            for k in ks:
                for metric in ['recall', 'ndcg', 'hit_rate']:
                    aggregated_results[k][metric] += run_results[k][metric]
            total_runs += 1
                
            info = f"    Results: "
            for k in ks:
                info += f"R@{k}={run_results[k]['recall']:.4f}, "
                info += f"NDCG@{k}={run_results[k]['ndcg']:.4f}, "
                info += f"HR@{k}={run_results[k]['hit_rate']:.4f} | "
            print(info)

        row = {"alpha_item": alpha_item}
        for k in ks:
            for metric in ['recall', 'ndcg', 'hit_rate']:
                avg_value = aggregated_results[k][metric] / total_runs if total_runs > 0 else 0
                row[f"{metric}@{k}"] = avg_value
            
        item_sim_output_rows.append(row)
        print(f"  Average results for alpha_item = {alpha_item}:")
        for k in ks:
            print(f"    @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    item_sim_df = pd.DataFrame(item_sim_output_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_item_sim = f"neumf_alpha_item_similarity_amazon_health_household.csv"
    item_sim_df.to_csv(csv_file_item_sim, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Item Similarity Evaluation complete in {elapsed_time/60:.2f} minutes. Results saved to '{csv_file_item_sim}'")
    print("\nPreview of Item Similarity results:")
    print(item_sim_df.to_string())
    
    # TIME-BASED COMMUNITY SIMILARITY
    print("\n" + "="*40)
    print("ABLATION 2: NEUMF WITH TIME-BASED COMMUNITY SIMILARITY")
    print("="*40 + "\n")
    
    user_item_timestamps, user_community, item_community = create_time_user_item_data(train_df)
    item_time_output_rows = []
    
    start_time = time.time()
    
    for alpha_time in tqdm(alpha_values, desc="Evaluating alpha_item values"):
        aggregated_results = {k: {metric: 0.0 for metric in ['recall', 'ndcg', 'hit_rate']} for k in ks}
        total_runs = 0
        
        print(f"\nEvaluating alpha_item = {alpha_time}")
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}")

            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            train_dataset = NeuMFDataset(train_interactions, n_items)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
   
            model = NeuMF(n_users=n_users, n_items=n_items, emb_dim=64)
            train_model(model, train_loader, epochs=15, verbose=False)
            
            run_results = evaluate_neumf_with_community_time(
                model, 
                test_interactions, 
                train_interactions, 
                user_item_timestamps, 
                user_community, 
                item_community,
                ks=ks,
                alpha_time=alpha_time
            )

            for k in ks:
                for metric in ['recall', 'ndcg', 'hit_rate']:
                    aggregated_results[k][metric] += run_results[k][metric]
            total_runs += 1

            info = f"    Results: "
            for k in ks:
                info += f"R@{k}={run_results[k]['recall']:.4f}, "
                info += f"NDCG@{k}={run_results[k]['ndcg']:.4f}, "
                info += f"HR@{k}={run_results[k]['hit_rate']:.4f} | "
            print(info)
        
        row = {"alpha_item": alpha_time}
        for k in ks:
            for metric in ['recall', 'ndcg', 'hit_rate']:
                avg_value = aggregated_results[k][metric] / total_runs if total_runs > 0 else 0
                row[f"{metric}@{k}"] = avg_value
            
        item_time_output_rows.append(row)

        print(f"  Average results for alpha_time = {alpha_time}:")
        for k in ks:
            print(f"    @K={k} → " +
                  f"Recall: {row[f'recall@{k}']:.4f}, " +
                  f"NDCG: {row[f'ndcg@{k}']:.4f}, " +
                  f"Hit Rate: {row[f'hit_rate@{k}']:.4f}")
    
    time_sim_df = pd.DataFrame(item_time_output_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_time_sim = f"neumf_alpha_time_similarity_amazon_health_household.csv"
    time_sim_df.to_csv(csv_file_time_sim, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Item Similarity Evaluation complete in {elapsed_time/60:.2f} minutes. Results saved to '{csv_file_time_sim}'")
    print("\nPreview of Item Similarity results:")
    print(time_sim_df.to_string())