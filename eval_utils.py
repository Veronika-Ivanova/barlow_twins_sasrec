import numpy as np
import torch
from sklearn.decomposition import PCA
import pandas as pd

from utils import topn_recommendations, downvote_seen_items
from data import data_to_sequences

def get_popularity_buckets(full_data, data_description, n_buckets=3):
    import pandas as pd

    userid = data_description['users']
    itemid = data_description['items']
    
    # Compute each item's popularity (number of unique users)
    item_popularity = (
        full_data.groupby(itemid)[userid]
        .nunique()
        .reset_index(name='popularity')
    )
    
    # Sort items by popularity (highest first)
    item_popularity = item_popularity.sort_values('popularity', ascending=False).reset_index(drop=True)
    
    total_pop = item_popularity['popularity'].sum()
    bucket_target = total_pop / n_buckets

    # Greedily assign buckets based on cumulative popularity
    cumulative = 0
    bucket = 1
    buckets = []
    for pop in item_popularity['popularity']:
        cumulative += pop
        buckets.append(bucket)
        # Only move to the next bucket if we haven’t reached the last one yet
        if bucket < n_buckets and cumulative >= bucket * bucket_target:
            bucket += 1

    item_popularity['bucket'] = buckets

    # Create a mapping from item to bucket
    item_to_bucket = dict(zip(item_popularity[itemid], item_popularity['bucket']))
    return item_to_bucket

def get_test_scores(model, data_description, testset_, holdout_, item_buckets, device):
    sasrec_scores = sasrec_model_scoring(model, testset_, data_description, device)
    downvote_seen_items(sasrec_scores, testset_, data_description)

    sasrec_recs = topn_recommendations(sasrec_scores, topn=50)
    test_scores = model_evaluate(sasrec_recs, holdout_, data_description)
    
    #Bucket based
    holdout_ = holdout_.assign(
        item_bucket = holdout_[data_description["items"]].map(item_buckets)
    )
    for i in range(1, holdout_["item_bucket"].max()+1):
        hld_filter = holdout_["item_bucket"] == i
        recs_bucket = sasrec_recs[ np.where(hld_filter.values)[0] ]
        test_scores_bi = model_evaluate(recs_bucket, holdout_[hld_filter], data_description)
        test_scores_bi = {f'Bucket_{i} {k}':v for k,v in test_scores_bi.items()}
        test_scores = test_scores | test_scores_bi
    
    return test_scores, sasrec_scores

def sasrec_model_scoring(params, data, data_description, device):
    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description, validation=True)
    # perform scoring on a user-batch level
    scores = []
    for _, seq in test_sequences:
        with torch.no_grad():
            predictions = model.score(torch.tensor(seq, device=device, dtype=torch.long))
        scores.append(predictions.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)

def fig_to_arr(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    canvas = FigureCanvas(fig)
    canvas.draw()
    dpi = fig.get_dpi()
    fig_width, fig_height = fig.get_size_inches()
    width = int(fig_width * dpi)
    height = int(fig_height * dpi)
    rgba_buffer = canvas.buffer_rgba()
    image = np.frombuffer(rgba_buffer, dtype='uint8').reshape(height, width, 4)
    image_np = image[..., :3]
    return image_np

def export_scatter(X_2d):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    kde = gaussian_kde(X_2d.T)
    density = kde(X_2d.T)
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=density, s=20, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Density')
    plt.tight_layout()

    image_np = fig_to_arr(fig)

    return image_np


def evaluate_embeddings(params, data, data_description, device, n_singular = 64):
    
    import umap

    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description, validation=True)
    test_sequences = test_sequences[:1000]
    embeddings = []
    for _, seq in test_sequences:
        with torch.no_grad():
             embedding = model.get_embeddings(torch.tensor(seq, device=device, dtype=torch.long))
        embeddings.append(embedding.detach().cpu())
    embeddings = torch.vstack(embeddings)
    embeddings_np = embeddings.cpu().numpy()

    #PCA Decomposition
    pca = PCA(n_components=n_singular)
    X_pca = pca.fit_transform(embeddings_np)
    singular_values = pca.singular_values_
    X_2d = X_pca[:, :2]

    #UMAP Decomposition
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d_umap = reducer.fit_transform(embeddings_np)

    #Add item emb
    item_embs = model.item_emb.weight.detach().cpu()
    A = item_embs.cpu() @ embeddings.T.cpu()
    A = A.detach().cpu().numpy()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    projection_items_2d = Vt[:2, :].T * s[:2]

    #Plot it
    image_np = export_scatter(X_2d_umap)
    image_np_item = export_scatter(projection_items_2d)

    #return singular_values, X_pca, X_2d_umap, embeddings, image_np
    return {
        "embeddings": embeddings,
        "item_embeddings": item_embs.numpy(),
        "seq_singular_values": singular_values,
        "logits_singular_values": s,
        "seq_pca": X_pca,
        "seq_umap": X_2d_umap,
        "item_pca": projection_items_2d,
        "image_umap": image_np,
        "image_item": image_np_item,
    }

def plot_user_item_relation_alt(seq_embeddings, item_embeddings, holdout_, data_description, neg_samples = 10, seed = 0):
    import matplotlib.pyplot as plt
    import umap.umap_ as umap
    from numpy.random import RandomState
    from sklearn.manifold import MDS

    #print(holdout_)
    #print(item_embeddings)

    rng = RandomState(seed)

    holdout_items = holdout_[data_description["items"]].values

    # Select positives
    n = len(holdout_)
    users_sample = rng.choice(np.arange(n), 5, replace=False)
    nn = len(users_sample)
    n_items = data_description['n_items']
    seq_embeddings = seq_embeddings[users_sample]
    pos_emb = item_embeddings[holdout_items][users_sample]
    neg_idx = np.concatenate([
        rng.choice(np.delete(np.arange(n_items), holdout_items[i]), neg_samples, replace=False)
        for i in np.arange(n)[users_sample]
    ])
    neg_emb = item_embeddings[neg_idx]

    combined = np.concatenate([seq_embeddings, pos_emb, neg_emb], axis=0)
    

    # Run UMAP to project to 2D.
    #umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(combined)
    
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    dot_matrix = np.dot(combined, combined.T)
    dot_matrix = np.sqrt( 1 - dot_matrix )
    mds = MDS(n_components=2, dissimilarity="precomputed")
    umap_2d = mds.fit_transform(dot_matrix)

    # Split the projected coordinates back into user, positive, and negative groups.
    U_2d = umap_2d[:nn]
    P_2d = umap_2d[nn:2*nn]
    N_2d = umap_2d[2*nn:]  # shape: (n*k, 2)

    # Create a colormap with n distinct colors.
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(nn)]

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot each user's embeddings along with their positive and negative items.
    for i in range(nn):
        # Plot user embedding as an "x"
        ax.scatter(U_2d[i, 0], U_2d[i, 1],
                marker='x', s=100,
                color=colors[i],
                label=f'User {i}')
        
        # Plot the positive embedding as a bright circle with a black edge.
        ax.scatter(P_2d[i, 0], P_2d[i, 1],
                marker='o', s=100,
                color=colors[i],
                edgecolor='k', linewidth=1.5)
        
        # Compute the slice for negative samples for this user.
        neg_start = i * neg_samples
        neg_end = (i + 1) * neg_samples
        ax.scatter(N_2d[neg_start:neg_end, 0],
                N_2d[neg_start:neg_end, 1],
                marker='o', s=100,
                color=colors[i],
                alpha=0.2)

    # Set title and axis labels.
    ax.set_title("UMAP Projection of Embeddings")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    # Add a legend.
    ax.legend(loc="best", fontsize=9)

    return fig_to_arr(fig)


def plot_user_item_relation(seq_embeddings, item_embeddings, holdout_, data_description, neg_samples = 300, num_users = 30, seed = 0):
    import matplotlib.pyplot as plt
    import umap.umap_ as umap
    from numpy.random import RandomState
    from sklearn.manifold import MDS
    from scipy.stats import gaussian_kde

    rng = RandomState(seed)

    holdout_items = holdout_[data_description["items"]].values
    holdout_items = holdout_items[:1000]

    scores = (seq_embeddings@item_embeddings.T)
    top_recs = topn_recommendations(scores)

    whr = np.where(top_recs == holdout_items.reshape(-1, 1))[0]

    good_users = whr
    print("Num good users for plot relations", len(good_users))

    # Select positives
    n = len(holdout_)
    users_sample = rng.choice(good_users, min(num_users, len(good_users)), replace=False)
    nn = len(users_sample)
    n_items = data_description['n_items']
    seq_embeddings = seq_embeddings[users_sample]
    pos_emb = item_embeddings[holdout_items][users_sample]
    neg_idx = np.concatenate([
        rng.choice(np.delete(np.arange(n_items), holdout_items[i]), neg_samples, replace=False)
        for i in np.arange(n)[users_sample]
    ])
    neg_emb = item_embeddings[neg_idx]
    x_positions = np.linspace(0, nn - 1, nn)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, x in enumerate(x_positions):
        user_embedding = seq_embeddings[i]
        
        # Compute distances for positives and negatives for this user
        pos_dists = np.dot(pos_emb[i], user_embedding)
        neg_dists = np.dot(neg_emb[i*neg_samples:(i+1)*neg_samples], user_embedding)
        
        # Add a small x-axis jitter for clarity
        jitter_range = 0.00
        x_jitter_pos = x + np.random.uniform(-jitter_range, jitter_range, size=pos_dists.shape)
        x_jitter_neg = x + np.random.uniform(-jitter_range, jitter_range, size=neg_dists.shape)
        
        # Plot negatives above y=0 (red) and positives below y=0 (blue)
        ax.scatter(x_jitter_neg[:20], neg_dists[:20], color='red', alpha=0.3,
                    label='Negatives' if i == 0 else "")
        kde_neg = gaussian_kde(neg_dists)
        y_neg = np.linspace(neg_dists.min(), neg_dists.max(), 100)
        density_neg = kde_neg(y_neg)
        density_neg = density_neg / density_neg.max() * 0.3
        ax.fill_betweenx(y_neg, x - density_neg, x + density_neg, color='red', alpha=0.2,
                         label='Negatives' if i == 0 else "")
        ax.scatter(x_jitter_pos, pos_dists, color='blue', alpha=0.7,
                    label='Positives' if i == 0 else "")
        
        # Mark the user embedding at y = 0
        ax.scatter(x, 0, color='green', marker='x', s=100,
                    label='User' if i == 0 else "")

    # Draw a horizontal line at y=0 to indicate user baseline
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Users")
    ax.set_ylabel("Item dot product")
    ax.legend()

    return fig_to_arr(fig)


def calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn):
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
   
    #NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1.)) / n_test_users

    #COV calculation
    cov = np.unique(recommended_items[:, :topn]).size / n_items

    #ILD calculation
    rec_sets = [set(recommended_items[i, :topn]) for i in range(n_test_users)]
    total_jaccard = 0.0
    count = 0
    for i in range(n_test_users):
        for j in range(i + 1, n_test_users):
            union = rec_sets[i] | rec_sets[j]
            if union:
                jaccard = len(rec_sets[i] & rec_sets[j]) / len(union)
            else:
                jaccard = 0.0
            total_jaccard += jaccard
            count += 1
    avg_jaccard = total_jaccard / count if count > 0 else 0.0
    ild = 1 - avg_jaccard

    return {'hr': hr, 'mrr': mrr, 'ndcg': ndcg, 'cov': cov, 'ild': ild}


def model_evaluate(recommended_items, holdout, holdout_description, topn_list=(1, 5, 10, 20, 50)):
    n_items = holdout_description['n_items']
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    n_test_users = recommended_items.shape[0]
    recommended_items
    assert recommended_items.shape[0] == len(holdout_items)

    metrics = {}
    for topn in topn_list:
        metrics = metrics | {f'{key}@{topn}': value for key, value in calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn).items()}

    return metrics
