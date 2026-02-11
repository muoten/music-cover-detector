#!/usr/bin/env python3
"""
Train an MLP to refine CoverHunter embeddings for better cover detection.
Uses cosine similarity on transformed embeddings with multi-negative contrastive loss
and online hard negative recomputation.

v2: MLP-based embedding refinement (vs v1's linear metric learning).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import argparse


# ============================================================================
# Model
# ============================================================================

class EmbeddingRefinerV2(nn.Module):
    """MLP that refines embeddings with residual connection.

    Output is L2-normalized so cosine similarity is the natural metric.
    Residual connection ensures the model starts near identity (preserving baseline).
    """

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = (input_dim == output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable gate: controls how much MLP output mixes with residual
        # Initialized near zero so model starts at identity
        self.gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ~ 0.05

    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            alpha = torch.sigmoid(self.gate)
            out = alpha * out + (1.0 - alpha) * x
        return F.normalize(out, p=2, dim=-1)


# ============================================================================
# Dataset
# ============================================================================

class RankingDataset(Dataset):
    """Dataset with multiple positives and semi-hard negatives per anchor.
    Supports online hard negative recomputation."""

    def __init__(self, embeddings, clique_ids, num_samples=100000,
                 num_positives=2, num_negatives=50):
        self.embeddings = torch.FloatTensor(embeddings)
        self.clique_ids = clique_ids
        self.num_samples = num_samples
        self.num_positives = num_positives
        self.num_negatives = num_negatives

        # Build clique -> indices mapping
        self.clique_to_indices = defaultdict(list)
        for idx, clique in enumerate(clique_ids):
            self.clique_to_indices[clique].append(idx)

        self.valid_cliques = [c for c, indices in self.clique_to_indices.items()
                              if len(indices) >= 2]
        self.all_indices = list(range(len(embeddings)))

        print(f"Valid cliques (>=2 songs): {len(self.valid_cliques)}")
        print(f"Total songs: {len(embeddings)}")
        print(f"Positives: {num_positives}, Negatives: {num_negatives}")

        self._recompute_neighbors(embeddings)

    def _recompute_neighbors(self, embeddings):
        nn_model = NearestNeighbors(n_neighbors=100, metric='cosine')
        nn_model.fit(embeddings)
        _, self.nearest_indices = nn_model.kneighbors(embeddings)

    def update_embeddings(self, new_embeddings):
        """Refresh hard negatives using current model's transformed embeddings."""
        print("  Recomputing hard negatives with current model...")
        self.embeddings = torch.FloatTensor(new_embeddings)
        self._recompute_neighbors(new_embeddings)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clique = random.choice(self.valid_cliques)
        indices = self.clique_to_indices[clique]

        # Anchor + positives from same clique
        selected = random.sample(indices, min(self.num_positives + 1, len(indices)))
        anchor_idx = selected[0]
        positive_indices = selected[1:]
        while len(positive_indices) < self.num_positives:
            positive_indices.append(random.choice(indices))

        # Semi-hard negatives: closest non-covers
        neg_indices = []
        for neg_idx in self.nearest_indices[anchor_idx]:
            if self.clique_ids[neg_idx] != clique and neg_idx not in neg_indices:
                neg_indices.append(neg_idx)
                if len(neg_indices) >= self.num_negatives:
                    break

        # Fill remaining with random negatives
        attempts = 0
        while len(neg_indices) < self.num_negatives and attempts < 200:
            neg_idx = random.choice(self.all_indices)
            if self.clique_ids[neg_idx] != clique and neg_idx not in neg_indices:
                neg_indices.append(neg_idx)
            attempts += 1
        while len(neg_indices) < self.num_negatives:
            neg_idx = random.choice(self.all_indices)
            if self.clique_ids[neg_idx] != clique:
                neg_indices.append(neg_idx)

        return (self.embeddings[anchor_idx],
                torch.stack([self.embeddings[i] for i in positive_indices]),
                torch.stack([self.embeddings[i] for i in neg_indices]))


# ============================================================================
# Loss
# ============================================================================

class MultiNegativeContrastiveLoss(nn.Module):
    """Cross-entropy over 1 positive + K negatives."""

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        anchor: (B, D), positive: (B, D), negatives: (B, K, D)
        """
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1), negatives, dim=2
        ) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, embeddings, clique_ids, device, batch_size=1000):
    """Compute Precision@1, Hit@5, MAP@5 on transformed embeddings."""
    model.eval()
    emb_tensor = torch.FloatTensor(embeddings).to(device)

    refined = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            refined.append(model(emb_tensor[i:i+batch_size]).cpu().numpy())
    refined = np.vstack(refined)

    n = len(refined)
    hits_at_1 = hits_at_5 = queries = 0
    ap_scores = []

    for batch_start in tqdm(range(0, n, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, n)
        distances = cosine_distances(refined[batch_start:batch_end], refined)

        for i, row in enumerate(distances):
            query_idx = batch_start + i
            query_clique = clique_ids[query_idx]
            n_covers = sum(1 for c in clique_ids if c == query_clique) - 1
            if n_covers == 0:
                continue
            queries += 1

            sorted_indices = np.argsort(row)
            sorted_indices = [idx for idx in sorted_indices if idx != query_idx][:100]

            if clique_ids[sorted_indices[0]] == query_clique:
                hits_at_1 += 1
            if query_clique in [clique_ids[idx] for idx in sorted_indices[:5]]:
                hits_at_5 += 1

            hits = ap = 0
            for k, idx in enumerate(sorted_indices[:5]):
                if clique_ids[idx] == query_clique:
                    hits += 1
                    ap += hits / (k + 1)
            ap_scores.append(ap / max(1, min(5, n_covers)))

    if queries == 0:
        return {'precision_at_1': 0.0, 'hit_at_5': 0.0, 'map_at_5': 0.0, 'num_queries': 0}

    return {
        'precision_at_1': hits_at_1 / queries,
        'hit_at_5': hits_at_5 / queries,
        'map_at_5': np.mean(ap_scores),
        'num_queries': queries,
    }


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading embeddings...")
    embeddings = np.loadtxt(args.embeddings, delimiter='\t')
    print(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    print("Loading metadata...")
    metadata = pd.read_csv(args.metadata, sep='\t')

    print("Loading ground truth...")
    ground_truth = pd.read_csv(args.ground_truth)
    ground_truth['youtube_id'] = ground_truth['youtube_url'].str.extract(r'v=([A-Za-z0-9_-]{11})')
    id_to_clique = dict(zip(ground_truth['youtube_id'], ground_truth['clique']))

    clique_ids = [id_to_clique.get(vid, f"unknown_{i}") for i, vid in enumerate(metadata['youtube_id'])]
    valid_indices = [i for i, c in enumerate(clique_ids) if not str(c).startswith('unknown_')]
    embeddings_valid = embeddings[valid_indices]
    clique_ids_valid = [clique_ids[i] for i in valid_indices]
    print(f"Songs with ground truth: {len(embeddings_valid)}")

    # Clique-aware train/val split
    if args.val_split > 0:
        clique_to_indices = defaultdict(list)
        for idx, clique in enumerate(clique_ids_valid):
            clique_to_indices[clique].append(idx)

        unique_cliques = list(clique_to_indices.keys())
        clique_train, clique_val = train_test_split(unique_cliques, test_size=args.val_split, random_state=42)

        train_indices = [i for c in clique_train for i in clique_to_indices[c]]
        val_indices = [i for c in clique_val for i in clique_to_indices[c]]

        embeddings_train = embeddings_valid[train_indices]
        clique_ids_train = [clique_ids_valid[i] for i in train_indices]
        embeddings_val = embeddings_valid[val_indices]
        clique_ids_val = [clique_ids_valid[i] for i in val_indices]
        print(f"Train: {len(embeddings_train)}, Val: {len(embeddings_val)} (clique-aware split)")
    else:
        embeddings_train = embeddings_valid
        clique_ids_train = clique_ids_valid
        embeddings_val = clique_ids_val = None

    # Baseline
    print("\n=== Baseline (cosine on raw embeddings) ===")
    baseline_model = nn.Identity().to(device)
    # Wrap Identity to add normalize (raw embeddings may not be normalized)
    class NormIdentity(nn.Module):
        def forward(self, x):
            return F.normalize(x, p=2, dim=-1)
    baseline_model = NormIdentity().to(device)

    train_baseline = evaluate(baseline_model, embeddings_train, clique_ids_train, device)
    print(f"Train P@1: {train_baseline['precision_at_1']*100:.1f}%  "
          f"Hit@5: {train_baseline['hit_at_5']*100:.1f}%  "
          f"MAP@5: {train_baseline['map_at_5']*100:.1f}%  "
          f"({train_baseline['num_queries']} queries)")

    if embeddings_val is not None:
        val_baseline = evaluate(baseline_model, embeddings_val, clique_ids_val, device)
        print(f"Val   P@1: {val_baseline['precision_at_1']*100:.1f}%  "
              f"Hit@5: {val_baseline['hit_at_5']*100:.1f}%  "
              f"MAP@5: {val_baseline['map_at_5']*100:.1f}%  "
              f"({val_baseline['num_queries']} queries)")
        baseline_metrics = val_baseline
        eval_embeddings, eval_clique_ids = embeddings_val, clique_ids_val
    else:
        baseline_metrics = train_baseline
        eval_embeddings, eval_clique_ids = embeddings_train, clique_ids_train

    # Model
    model = EmbeddingRefinerV2(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel: EmbeddingRefinerV2 ({sum(p.numel() for p in model.parameters())} params)")
    print(f"  {embeddings.shape[1]} -> {args.hidden_dim} -> {args.output_dim} + residual + L2-norm")

    # Dataset & loss
    dataset = RankingDataset(
        embeddings_train, clique_ids_train,
        num_samples=args.triplets_per_epoch,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    criterion = MultiNegativeContrastiveLoss(temperature=args.temperature)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_precision = baseline_metrics['precision_at_1']
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    prev_loss = float('inf')

    print(f"\n=== Training ===")
    print(f"lr={args.lr}, weight_decay={args.weight_decay}, temperature={args.temperature}")
    print(f"dropout={args.dropout}, epochs={args.epochs}, batch_size={args.batch_size}")

    for epoch in range(args.epochs):
        # Online hard negative recomputation
        if epoch > 0:
            model.eval()
            with torch.no_grad():
                train_tensor = torch.FloatTensor(embeddings_train).to(device)
                transformed = []
                for i in range(0, len(embeddings_train), 1000):
                    transformed.append(model(train_tensor[i:i+1000]).cpu().numpy())
                transformed = np.vstack(transformed)
            dataset.update_embeddings(transformed)

        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for anchor, positives, negatives in pbar:
            anchor = anchor.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            B, num_pos, D = positives.shape
            num_neg = negatives.size(1)

            anchor_out = model(anchor)
            positives_out = model(positives.reshape(-1, D)).view(B, num_pos, -1)
            negatives_out = model(negatives.reshape(-1, D)).view(B, num_neg, -1)

            # Select best positive (highest cosine sim to anchor)
            pos_sims = F.cosine_similarity(anchor_out.unsqueeze(1), positives_out, dim=2)
            best_pos_idx = pos_sims.argmax(dim=1)
            best_positive = positives_out[torch.arange(B), best_pos_idx]

            loss = criterion(anchor_out, best_positive, negatives_out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        loss_trend = ">" if epoch == 0 else ("v" if avg_loss < prev_loss else "^")

        # Evaluate
        if embeddings_val is not None:
            metrics = evaluate(model, embeddings_val, clique_ids_val, device)
            eval_set = "Val"
        else:
            metrics = evaluate(model, embeddings_train, clique_ids_train, device)
            eval_set = "Train"

        # Embedding health check (every 2 epochs)
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample = model(torch.FloatTensor(embeddings_train[:100]).to(device))
                pw_sim = F.cosine_similarity(sample.unsqueeze(1), sample.unsqueeze(0), dim=2)
                mask = ~torch.eye(len(sample), dtype=torch.bool, device=device)
                avg_sim = pw_sim[mask].mean().item()
                gate_val = torch.sigmoid(model.gate).item()
                print(f"  gate={gate_val:.3f} (MLP mix), avg_sim={avg_sim:.3f}")
                if avg_sim > 0.95:
                    print(f"  Warning: embeddings collapsing!")

        print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f} {loss_trend}, "
              f"{eval_set} P@1={metrics['precision_at_1']*100:.1f}%, "
              f"Hit@5={metrics['hit_at_5']*100:.1f}%, "
              f"MAP@5={metrics['map_at_5']*100:.1f}%")

        prev_loss = avg_loss

        # Track best
        if metrics['precision_at_1'] > best_precision:
            best_precision = metrics['precision_at_1']
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  New best! P@1={best_precision*100:.1f}%")
        else:
            patience_counter += 1
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"  Early stopping after {patience_counter} epochs without improvement")
                break

    # Restore best and save
    model.load_state_dict(best_model_state)

    best_metrics = evaluate(model, eval_embeddings, eval_clique_ids, device)
    if best_metrics['precision_at_1'] >= baseline_metrics['precision_at_1']:
        torch.save(best_model_state, args.output_model)
        print(f"\nSaved model to {args.output_model}")
    else:
        print(f"\nModel did not improve over baseline. Not saved.")

    # Generate refined embeddings for all songs
    print("Generating refined embeddings for all songs...")
    model.eval()
    emb_tensor = torch.FloatTensor(embeddings).to(device)
    refined_all = []
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), 1000)):
            refined_all.append(model(emb_tensor[i:i+1000]).cpu().numpy())
    refined_all = np.vstack(refined_all)
    np.savetxt(args.output_embeddings, refined_all, delimiter='\t', fmt='%.6f')
    print(f"Saved refined embeddings to {args.output_embeddings}")

    # Final results
    print("\n=== Results ===")
    final_p1 = best_metrics['precision_at_1'] * 100
    base_p1 = baseline_metrics['precision_at_1'] * 100
    print(f"Baseline P@1:  {base_p1:.1f}%")
    print(f"Refined P@1:   {final_p1:.1f}%")
    print(f"Improvement:   {final_p1 - base_p1:+.1f}%")
    print(f"Best epoch:    {args.epochs - patience_counter}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP embedding refiner (v2) for cover detection')

    # Data
    parser.add_argument('--embeddings', default='vectors_without_metadata_v5.tsv')
    parser.add_argument('--metadata', default='metadata_clean_v5.tsv')
    parser.add_argument('--ground-truth', default='videos_to_test.csv')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=256, help='MLP hidden dimension')
    parser.add_argument('--output-dim', type=int, default=128, help='Output embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--triplets-per-epoch', type=int, default=100000)

    # Loss
    parser.add_argument('--temperature', type=float, default=0.2, help='Contrastive loss temperature')
    parser.add_argument('--num-positives', type=int, default=2)
    parser.add_argument('--num-negatives', type=int, default=50)

    # Validation
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--early-stopping', type=int, default=5)

    # Output
    parser.add_argument('--output-model', default='embedding_refiner_v2.pt')
    parser.add_argument('--output-embeddings', default='vectors_refined_v2.tsv')

    args = parser.parse_args()
    train(args)
