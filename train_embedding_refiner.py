#!/usr/bin/env python3
"""
Train a projection head on frozen CoverHunter embeddings to improve cover detection.
Uses verified cover pairs from Discogs as supervision.
Optimized for precision@1 metric in cover discovery task.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import argparse

# ============================================================================
# Model
# ============================================================================

class EmbeddingRefiner(nn.Module):
    """MLP that refines embeddings for better cover detection."""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, use_linear=False):
        super().__init__()
        self.use_linear = use_linear

        if use_linear:
            # Simple linear projection - less prone to overfitting
            self.mlp = nn.Linear(input_dim, output_dim, bias=False)
            # Initialize very close to identity to preserve baseline performance
            if input_dim == output_dim:
                with torch.no_grad():
                    # Start very close to identity (99% identity, 1% noise)
                    identity = torch.eye(input_dim)
                    noise = 0.01 * torch.randn(input_dim, input_dim)
                    self.mlp.weight.data = identity + noise
            else:
                # For different dimensions, use Xavier but scaled down
                nn.init.xavier_uniform_(self.mlp.weight, gain=0.1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=1)  # L2 normalize output


class MetricLearner(nn.Module):
    """Learn a distance metric optimized for cover detection.
    Learns a linear transform W and computes similarity as cosine(Wx, Wy).
    This keeps similarity bounded in [-1, 1] and distance in [0, 2].

    For 'mahalanobis' mode: d(x,y) = ||W(x-y)||^2 (Mahalanobis-like)
    For 'bilinear' mode: sim(x,y) = cosine(Wx, Wy) (normalized bilinear)
    """

    def __init__(self, input_dim=128, metric_type='mahalanobis', rank=None):
        super().__init__()
        self.metric_type = metric_type
        self.input_dim = input_dim

        if rank is None:
            rank = input_dim
        rank = min(rank, input_dim)

        self.W = nn.Parameter(torch.empty(rank, input_dim))
        with torch.no_grad():
            if rank == input_dim:
                # Initialize near identity: preserves cosine similarity at start
                self.W.data = torch.eye(input_dim) + 0.01 * torch.randn(input_dim, input_dim)
            else:
                # Low-rank: first rank dimensions of identity
                self.W.data = torch.eye(rank, input_dim) + 0.01 * torch.randn(rank, input_dim)

    def _transform(self, x):
        """Apply learned transform and L2-normalize."""
        return F.normalize(torch.matmul(x, self.W.T), p=2, dim=-1)

    def compute_similarity(self, x, y):
        """Compute learned similarity in [-1, 1]."""
        return (self._transform(x) * self._transform(y)).sum(dim=-1)

    def compute_distance(self, x, y):
        """Compute learned distance in [0, 2]."""
        return 1.0 - self.compute_similarity(x, y)

    def forward(self, x, y=None):
        """
        If y is None, returns x (for compatibility with evaluation).
        If y is provided, returns learned distance/similarity.
        """
        if y is None:
            # For evaluation compatibility, just return embeddings as-is
            return x
        else:
            return self.compute_similarity(x, y)


# ============================================================================
# Dataset
# ============================================================================

class TripletDataset(Dataset):
    """Dataset that generates triplets: (anchor, positive, negative)."""

    def __init__(self, embeddings, clique_ids, num_triplets=100000, hard_negatives=False):
        self.embeddings = torch.FloatTensor(embeddings)
        self.embeddings_np = embeddings
        self.clique_ids = clique_ids
        self.num_triplets = num_triplets
        self.hard_negatives = hard_negatives

        # Build clique -> indices mapping
        self.clique_to_indices = {}
        for idx, clique in enumerate(clique_ids):
            if clique not in self.clique_to_indices:
                self.clique_to_indices[clique] = []
            self.clique_to_indices[clique].append(idx)

        # Filter to cliques with at least 2 songs (can form positive pairs)
        self.valid_cliques = [c for c, indices in self.clique_to_indices.items()
                             if len(indices) >= 2]
        self.all_indices = list(range(len(embeddings)))

        print(f"Valid cliques (≥2 songs): {len(self.valid_cliques)}")
        print(f"Total songs: {len(embeddings)}")

        # Precompute nearest neighbors for hard negative mining
        if hard_negatives:
            print("Precomputing nearest neighbors for hard negatives...")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=50, metric='cosine')
            nn.fit(embeddings)
            _, self.nearest_indices = nn.kneighbors(embeddings)
            print("Done.")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # Random anchor from a valid clique
        clique = random.choice(self.valid_cliques)
        indices = self.clique_to_indices[clique]

        # Anchor and positive from same clique
        anchor_idx, positive_idx = random.sample(indices, 2)

        if self.hard_negatives:
            # Hard negative: closest song from a different clique
            for neg_idx in self.nearest_indices[anchor_idx]:
                if self.clique_ids[neg_idx] != clique:
                    negative_idx = neg_idx
                    break
            else:
                # Fallback to random if no hard negative found
                negative_idx = anchor_idx
                while self.clique_ids[negative_idx] == clique:
                    negative_idx = random.choice(self.all_indices)
        else:
            # Random negative from different clique
            negative_idx = anchor_idx
            while self.clique_ids[negative_idx] == clique:
                negative_idx = random.choice(self.all_indices)

        return (self.embeddings[anchor_idx],
                self.embeddings[positive_idx],
                self.embeddings[negative_idx])


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with multiple hard negatives."""

    def __init__(self, embeddings, clique_ids, num_samples=100000, num_hard_negatives=5, hard_ratio=1.0):
        self.embeddings = torch.FloatTensor(embeddings)
        self.clique_ids = clique_ids
        self.num_samples = num_samples
        self.num_hard_negatives = num_hard_negatives
        self.hard_ratio = hard_ratio  # Fraction of negatives that are hard (rest are random)

        # Build clique -> indices mapping
        self.clique_to_indices = {}
        for idx, clique in enumerate(clique_ids):
            if clique not in self.clique_to_indices:
                self.clique_to_indices[clique] = []
            self.clique_to_indices[clique].append(idx)

        # Filter to cliques with at least 2 songs
        self.valid_cliques = [c for c, indices in self.clique_to_indices.items()
                             if len(indices) >= 2]
        self.all_indices = list(range(len(embeddings)))

        print(f"Valid cliques (≥2 songs): {len(self.valid_cliques)}")
        print(f"Total songs: {len(embeddings)}")
        print(f"Negatives per sample: {num_hard_negatives} ({hard_ratio*100:.0f}% hard, {(1-hard_ratio)*100:.0f}% random)")

        # Precompute nearest neighbors
        print("Precomputing nearest neighbors...")
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=50, metric='cosine')
        nn.fit(embeddings)
        _, self.nearest_indices = nn.kneighbors(embeddings)
        print("Done.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random anchor from a valid clique
        clique = random.choice(self.valid_cliques)
        indices = self.clique_to_indices[clique]

        # Anchor and positive from same clique
        anchor_idx, positive_idx = random.sample(indices, 2)

        # Calculate how many hard vs random negatives
        num_hard = int(self.num_hard_negatives * self.hard_ratio)
        num_random = self.num_hard_negatives - num_hard

        neg_indices = []

        # Get hard negatives (closest non-covers)
        for neg_idx in self.nearest_indices[anchor_idx]:
            if self.clique_ids[neg_idx] != clique:
                neg_indices.append(neg_idx)
                if len(neg_indices) >= num_hard:
                    break

        # Add random negatives
        attempts = 0
        while len(neg_indices) < self.num_hard_negatives and attempts < 100:
            neg_idx = random.choice(self.all_indices)
            if self.clique_ids[neg_idx] != clique and neg_idx not in neg_indices:
                neg_indices.append(neg_idx)
            attempts += 1

        # Pad with any negatives if needed
        while len(neg_indices) < self.num_hard_negatives:
            neg_idx = random.choice(self.all_indices)
            if self.clique_ids[neg_idx] != clique:
                neg_indices.append(neg_idx)

        return (self.embeddings[anchor_idx],
                self.embeddings[positive_idx],
                torch.stack([self.embeddings[i] for i in neg_indices]))


class RankingDataset(Dataset):
    """Dataset for ranking-based learning with multiple positives and negatives.
    Optimized for precision@1: provides anchor, multiple positives, and negatives
    that are likely to be confused (top-k but wrong)."""

    def __init__(self, embeddings, clique_ids, num_samples=100000, 
                 num_positives=2, num_negatives=10, use_semi_hard=True):
        self.embeddings = torch.FloatTensor(embeddings)
        self.embeddings_np = embeddings
        self.clique_ids = clique_ids
        self.num_samples = num_samples
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.use_semi_hard = use_semi_hard

        # Build clique -> indices mapping
        self.clique_to_indices = {}
        for idx, clique in enumerate(clique_ids):
            if clique not in self.clique_to_indices:
                self.clique_to_indices[clique] = []
            self.clique_to_indices[clique].append(idx)

        # Filter to cliques with at least 2 songs
        self.valid_cliques = [c for c, indices in self.clique_to_indices.items()
                             if len(indices) >= 2]
        self.all_indices = list(range(len(embeddings)))

        print(f"Valid cliques (≥2 songs): {len(self.valid_cliques)}")
        print(f"Total songs: {len(embeddings)}")
        print(f"Positives per sample: {num_positives}, Negatives: {num_negatives}")

        # Precompute nearest neighbors for semi-hard negative mining
        if use_semi_hard:
            self._recompute_neighbors(embeddings)

    def _recompute_neighbors(self, embeddings):
        """Recompute nearest neighbors from given embeddings."""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=100, metric='cosine')
        nn.fit(embeddings)
        _, self.nearest_indices = nn.kneighbors(embeddings)

    def update_embeddings(self, new_embeddings):
        """Refresh hard negatives using transformed embeddings from current model."""
        print("  Recomputing hard negatives with current model...")
        self.embeddings = torch.FloatTensor(new_embeddings)
        if self.use_semi_hard:
            self._recompute_neighbors(new_embeddings)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random anchor from a valid clique
        clique = random.choice(self.valid_cliques)
        indices = self.clique_to_indices[clique]

        # Anchor and positives from same clique
        selected = random.sample(indices, min(self.num_positives + 1, len(indices)))
        anchor_idx = selected[0]
        positive_indices = selected[1:]

        # If we need more positives, sample with replacement
        while len(positive_indices) < self.num_positives:
            positive_indices.append(random.choice(indices))

        # Get negatives - focus on semi-hard (confusing) negatives
        neg_indices = []
        
        if self.use_semi_hard:
            # Semi-hard negatives: closest non-covers (likely to be confused)
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

        # Pad if needed
        while len(neg_indices) < self.num_negatives:
            neg_idx = random.choice(self.all_indices)
            if self.clique_ids[neg_idx] != clique:
                neg_indices.append(neg_idx)

        return (self.embeddings[anchor_idx],
                torch.stack([self.embeddings[i] for i in positive_indices]),
                torch.stack([self.embeddings[i] for i in neg_indices]))


# ============================================================================
# Loss Functions
# ============================================================================

class TripletLoss(nn.Module):
    """Triplet loss with hard margin."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR-style)."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Stack positive and negative
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature

        # Contrastive loss: positive should be more similar
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class MultiNegativeContrastiveLoss(nn.Module):
    """Contrastive loss with multiple hard negatives per anchor."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positive: (batch_size, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # Positive similarity: (batch_size,)
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature

        # Negative similarities: (batch_size, num_negatives)
        # anchor: (batch_size, 1, embed_dim), negatives: (batch_size, num_negatives, embed_dim)
        anchor_expanded = anchor.unsqueeze(1)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2) / self.temperature

        # Combine: positive is index 0, negatives are indices 1 to num_negatives
        # logits: (batch_size, 1 + num_negatives)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)


class InBatchNTXentLoss(nn.Module):
    """In-batch contrastive loss where all other samples are negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, _negatives=None):
        """
        Uses all other samples in batch as negatives.
        Args:
            anchor: (batch_size, embed_dim)
            positive: (batch_size, embed_dim)
            _negatives: ignored, uses in-batch negatives instead
        """
        batch_size = anchor.size(0)

        # Compute all pairwise similarities
        # anchor @ positive.T gives (batch_size, batch_size) similarity matrix
        sim_matrix = torch.mm(anchor, positive.T) / self.temperature

        # Diagonal elements are positive pairs
        # Off-diagonal elements are negative pairs
        labels = torch.arange(batch_size, device=anchor.device)

        # Cross entropy treats each row as a classification problem
        # where the correct class is the corresponding positive
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class RankingLoss(nn.Module):
    """Ranking loss optimized for precision@1.
    Penalizes when negatives rank above positives in the similarity ranking."""

    def __init__(self, margin=0.3, top_k_weight=3.0, pos_weight=0.5):
        """
        Args:
            margin: Minimum margin between positive and negative similarities
            top_k_weight: Extra weight for top-k negatives (most confusing ones)
            pos_weight: Weight for maximizing positive similarity (0 = disabled)
        """
        super().__init__()
        self.margin = margin
        self.top_k_weight = top_k_weight
        self.pos_weight = pos_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities: (batch_size, num_pos/num_neg)
        anchor_expanded = anchor.unsqueeze(1)  # (batch_size, 1, embed_dim)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # For each positive, ensure it ranks above all negatives
        # Expand to compare each positive with each negative
        pos_sim_expanded = pos_sim.unsqueeze(2)  # (batch_size, num_pos, 1)
        neg_sim_expanded = neg_sim.unsqueeze(1)  # (batch_size, 1, num_neg)

        # Margin ranking: positive should be at least margin more similar
        # loss = max(0, margin - (pos_sim - neg_sim))
        margin_loss = F.relu(self.margin - (pos_sim_expanded - neg_sim_expanded))
        
        # Weight top-k most confusing negatives more heavily
        # Sort negatives by similarity (most similar = most confusing)
        neg_sim_sorted, _ = torch.sort(neg_sim, dim=1, descending=True)
        top_k = min(5, num_neg)
        top_k_threshold = neg_sim_sorted[:, top_k - 1:top_k]  # (batch_size, 1)
        
        # Create weight matrix: higher weight for top-k negatives
        weights = torch.ones_like(neg_sim_expanded)
        top_k_mask = neg_sim_expanded >= top_k_threshold.unsqueeze(1)
        weights[top_k_mask] = self.top_k_weight
        
        weighted_loss = margin_loss * weights
        
        # Also maximize positive similarity (pull positives closer)
        pos_loss = 0.0
        if self.pos_weight > 0:
            # Average positive similarity (we want to maximize it)
            avg_pos_sim = pos_sim.mean()
            # Loss = 1 - similarity (since similarity is in [-1, 1], we want it close to 1)
            pos_loss = self.pos_weight * (1.0 - avg_pos_sim)
        
        return weighted_loss.mean() + pos_loss


class ApproxNDCGLoss(nn.Module):
    """Approximate NDCG loss for ranking optimization.
    Directly optimizes ranking quality, which correlates with precision@1."""

    def __init__(self, temperature=1.0, top_k=10):
        """
        Args:
            temperature: Temperature for softmax approximation
            top_k: Focus on top-k ranking positions
        """
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Combine positives and negatives
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        
        # Create relevance labels: 1 for positives, 0 for negatives
        relevance = torch.cat([
            torch.ones(batch_size, num_pos, device=anchor.device),
            torch.zeros(batch_size, num_neg, device=anchor.device)
        ], dim=1)

        # Approximate ranking with softmax
        # Higher similarity -> higher rank
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)

        # DCG approximation: sum of (relevance * prob) / log2(rank + 1)
        # For simplicity, use position weights
        num_items = num_pos + num_neg
        positions = torch.arange(1, num_items + 1, device=anchor.device, dtype=torch.float32)
        position_weights = 1.0 / torch.log2(positions + 1)
        
        # Weighted relevance
        weighted_relevance = relevance * position_weights.unsqueeze(0)
        
        # Expected DCG
        expected_dcg = (probs * weighted_relevance).sum(dim=1)
        
        # Ideal DCG (if all positives ranked first)
        ideal_dcg = (relevance * position_weights.unsqueeze(0)).sum(dim=1)
        
        # NDCG = DCG / IDCG
        ndcg = expected_dcg / (ideal_dcg + 1e-8)
        
        # Loss = 1 - NDCG (we want to maximize NDCG)
        loss = 1.0 - ndcg.mean()
        
        return loss


class PrecisionAt1Loss(nn.Module):
    """Direct optimization for precision@1.
    Maximizes the probability that the top-ranked item is a positive."""

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature for softmax (lower = sharper distribution)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Get best positive similarity (most similar positive)
        best_pos_sim, _ = torch.max(pos_sim, dim=1)  # (batch_size,)

        # Combine best positive with all negatives
        # We want the best positive to rank #1
        all_sim = torch.cat([best_pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_neg)
        
        # Softmax to get ranking probabilities
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)
        
        # Probability that positive is ranked #1
        pos_prob = probs[:, 0]
        
        # Loss = negative log probability (we want to maximize probability)
        loss = -torch.log(pos_prob + 1e-8).mean()
        
        return loss


class DirectPAt1Loss(nn.Module):
    """More direct P@1 optimization.
    Key insight: P@1 requires the CLOSEST item to be a positive.
    This loss directly optimizes: max(positive_similarities) > max(negative_similarities).
    It also uses a ranking-based approach that considers all items together."""

    def __init__(self, temperature=0.1, margin=0.1, hard_negative_weight=2.0):
        """
        Args:
            temperature: Temperature for softmax
            margin: Minimum margin between best positive and best negative
            hard_negative_weight: Extra weight for hard negatives (closest negatives)
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Get best positive and best negative similarities
        best_pos_sim, _ = torch.max(pos_sim, dim=1)  # (batch_size,)
        best_neg_sim, _ = torch.max(neg_sim, dim=1)  # (batch_size,)

        # Loss 1: Ensure best positive is better than best negative (with margin)
        # This directly optimizes for P@1: closest item should be a positive
        # Use a smoother loss that provides gradients even when margin is satisfied
        similarity_diff = best_pos_sim - best_neg_sim
        margin_loss = F.relu(self.margin - similarity_diff)
        
        # Also add a term to maximize the margin (not just satisfy it)
        # This helps push positives further from negatives
        margin_enhancement = -0.1 * similarity_diff.mean()  # Encourage larger margin
        
        # Loss 2: Ranking loss - but only for hard cases (negatives that are too close)
        # Don't penalize all negatives, only those that are problematic
        pos_sim_expanded = pos_sim.unsqueeze(2)  # (batch_size, num_pos, 1)
        neg_sim_expanded = neg_sim.unsqueeze(1)  # (batch_size, 1, num_neg)
        
        # Only penalize negatives that are closer than positives (hard negatives)
        ranking_loss = F.relu(self.margin - (pos_sim_expanded - neg_sim_expanded))
        
        # Weight hard negatives more (negatives that are close to positives)
        # Sort negatives by similarity
        neg_sim_sorted, _ = torch.sort(neg_sim, dim=1, descending=True)
        top_k = min(3, num_neg)  # Top 3 hardest negatives
        if top_k > 0:
            top_k_threshold = neg_sim_sorted[:, top_k - 1:top_k]  # (batch_size, 1)
            weights = torch.ones_like(neg_sim_expanded)
            hard_mask = neg_sim_expanded >= top_k_threshold.unsqueeze(1)
            weights[hard_mask] = self.hard_negative_weight
            ranking_loss = ranking_loss * weights
        
        # Combine losses - weight margin loss more heavily (it's more important for P@1)
        total_loss = 2.0 * margin_loss.mean() + margin_enhancement + 0.5 * ranking_loss.mean()
        
        return total_loss


class ListNetLoss(nn.Module):
    """ListNet loss - directly optimizes ranking quality.
    Uses permutation probability to model ranking, which is closer to P@1."""

    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Combine all items
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        
        # Create relevance scores: 1 for positives, 0 for negatives
        relevance = torch.cat([
            torch.ones(batch_size, num_pos, device=anchor.device),
            torch.zeros(batch_size, num_neg, device=anchor.device)
        ], dim=1)  # (batch_size, num_pos + num_neg)

        # ListNet: Use top-1 probability
        # The probability that the top-ranked item is relevant (positive)
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)  # (batch_size, num_pos + num_neg)
        
        # Expected relevance at position 1 = sum of (relevance * prob of being #1)
        # But we can approximate: probability that top-1 is a positive
        # This is the sum of probabilities of all positives being ranked #1
        pos_probs = probs[:, :num_pos]  # Probabilities of positives
        pos_at_1_prob = pos_probs.sum(dim=1)  # Probability that any positive is #1
        
        # Loss = negative log of probability that top-1 is positive
        loss = -torch.log(pos_at_1_prob + 1e-8).mean()
        
        return loss


class HitAtKLoss(nn.Module):
    """Optimize for Hit@K - probability that at least one positive appears in top-K.
    
    This is more forgiving than P@1 and often easier to optimize, while still
    being a strong proxy for improving P@1. If we can get positives into top-3,
    we're much closer to getting them to #1.
    
    Key insight: Instead of requiring the #1 position, we maximize the probability
    that at least one positive appears in the top-K ranked items.
    """

    def __init__(self, k=3, temperature=0.1, pos_weight=0.5):
        """
        Args:
            k: Target top-K positions (default: 3 for Hit@3)
            temperature: Temperature for softmax (lower = sharper ranking)
            pos_weight: Weight for maximizing positive similarities
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.pos_weight = pos_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Combine all similarities: positives and negatives
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        
        # Get ranking probabilities using softmax
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)  # (batch_size, num_pos + num_neg)
        
        # Sort by probability (descending) to get top-K
        probs_sorted, indices_sorted = torch.sort(probs, dim=1, descending=True)
        
        # Get top-K probabilities and indices
        top_k_probs = probs_sorted[:, :self.k]  # (batch_size, k)
        top_k_indices = indices_sorted[:, :self.k]  # (batch_size, k)
        
        # Check which top-K items are positives (indices < num_pos)
        is_positive = top_k_indices < num_pos  # (batch_size, k)
        
        # Probability that at least one positive is in top-K
        # Sum probabilities of positive items that appear in top-K
        pos_in_topk_probs = torch.zeros(batch_size, device=anchor.device)
        for i in range(batch_size):
            pos_mask = is_positive[i]  # (k,)
            if pos_mask.any():
                pos_in_topk_probs[i] = top_k_probs[i][pos_mask].sum()
            else:
                # No positives in top-K - penalty
                pos_in_topk_probs[i] = 0.0
        
        # Also compute weighted probability based on ranks of positives
        # Get the rank of each positive by finding where each positive appears in sorted order
        all_sim_sorted, all_indices_sorted = torch.sort(all_sim, dim=1, descending=True)
        
        # For each positive, find its rank
        pos_ranks = torch.zeros(batch_size, num_pos, device=anchor.device)
        for i in range(batch_size):
            for j in range(num_pos):
                # Find where positive j appears in the sorted list
                matches = (all_indices_sorted[i] == j).nonzero(as_tuple=True)[0]
                if matches.numel() > 0:
                    pos_ranks[i, j] = matches[0].float() + 1.0  # 1-indexed rank
                else:
                    pos_ranks[i, j] = num_pos + num_neg + 1.0  # Worst rank if not found
        
        # Weight by inverse rank (higher weight for better ranks, especially top-K)
        rank_weights = 1.0 / (pos_ranks + 1.0)  # (batch_size, num_pos)
        top_k_mask = pos_ranks <= self.k
        rank_weights[top_k_mask] *= (self.k + 1.0)  # Boost top-K positions
        
        # Weighted probability: sum of (prob * rank_weight) for positives
        pos_probs = probs[:, :num_pos]  # (batch_size, num_pos)
        weighted_pos_prob = (pos_probs * rank_weights).sum(dim=1)  # (batch_size,)
        
        # Combine direct hit probability + weighted probability
        # Direct hit is more important (it's what we're optimizing for)
        hit_prob = 0.7 * pos_in_topk_probs + 0.3 * weighted_pos_prob
        
        # Loss = negative log probability (we want to maximize probability)
        loss = -torch.log(hit_prob + 1e-8).mean()
        
        # Also add a term to maximize positive similarities (pull positives closer)
        if self.pos_weight > 0:
            avg_pos_sim = pos_sim.mean()
            pos_loss = self.pos_weight * (1.0 - avg_pos_sim)
            loss = loss + pos_loss
        
        return loss


class ReciprocalRankLoss(nn.Module):
    """Optimize for Mean Reciprocal Rank (MRR) - the reciprocal of the rank of the first positive.
    
    This is often better than Hit@K because:
    1. It directly optimizes for the rank of the first positive (closer to P@1)
    2. It's differentiable and provides smooth gradients
    3. It heavily penalizes when positives are ranked low, but rewards high ranks exponentially
    
    Key insight: MRR = 1/rank_of_first_positive. If first positive is at rank 1, MRR=1.0.
    If at rank 3, MRR=0.33. This naturally focuses on getting positives to the top.
    """

    def __init__(self, temperature=0.1, pos_weight=0.3):
        """
        Args:
            temperature: Temperature for softmax (lower = sharper ranking)
            pos_weight: Weight for maximizing positive similarities
        """
        super().__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Combine all similarities
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        
        # Get ranking probabilities using softmax
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)  # (batch_size, num_pos + num_neg)
        
        # Compute expected reciprocal rank
        # For each position, compute: prob(position) * (1 / position)
        # Sum over all positions where positives appear
        positions = torch.arange(1, num_pos + num_neg + 1, device=anchor.device, dtype=torch.float32)
        reciprocal_ranks = 1.0 / positions  # (num_pos + num_neg,)
        
        # Expected MRR = sum over all items of: prob(item is ranked at position i) * (1/i)
        # But we only care about positives, so we weight by whether it's a positive
        # More directly: for each positive, compute expected reciprocal rank
        pos_probs = probs[:, :num_pos]  # (batch_size, num_pos)
        
        # For each positive, compute its expected reciprocal rank
        # We need to find the expected rank of each positive
        # Sort all items by similarity to get ranks
        all_sim_sorted, all_indices_sorted = torch.sort(all_sim, dim=1, descending=True)
        
        # Compute expected reciprocal rank more efficiently
        # For each batch item, we want: E[1/rank_of_first_positive]
        # We can approximate this by:
        # 1. Sort items by probability to get expected ranks
        # 2. For each positive, compute: prob(positive) * (1 / expected_rank)
        
        # Get position weights (reciprocal ranks)
        num_items = num_pos + num_neg
        reciprocal_ranks = 1.0 / torch.arange(1, num_items + 1, device=anchor.device, dtype=torch.float32)  # (num_items,)
        
        # Sort probabilities to get expected ranking
        probs_sorted, indices_sorted = torch.sort(probs, dim=1, descending=True)  # (batch_size, num_items)
        
        # For each positive, find its expected rank and compute weighted reciprocal rank
        expected_rr = torch.zeros(batch_size, device=anchor.device)
        for j in range(num_pos):
            # Find where positive j appears in sorted list for each batch item
            for i in range(batch_size):
                pos_rank = (indices_sorted[i] == j).nonzero(as_tuple=True)[0]
                if pos_rank.numel() > 0:
                    rank_idx = pos_rank[0].item()
                    # Weight by probability of positive and reciprocal rank
                    expected_rr[i] += probs[i, j] * reciprocal_ranks[rank_idx]
        
        # Loss = negative expected reciprocal rank (we want to maximize MRR)
        loss = -expected_rr.mean()
        
        # Also add a term to maximize positive similarities
        if self.pos_weight > 0:
            avg_pos_sim = pos_sim.mean()
            pos_loss = self.pos_weight * (1.0 - avg_pos_sim)
            loss = loss + pos_loss
        
        return loss


class TopKMarginLoss(nn.Module):
    """Top-K Margin Loss: Ensures positives in top-K have a margin over ALL negatives.
    
    This is better than Hit@K because:
    1. It doesn't just ensure positives are in top-K, but that they're WELL-SEPARATED
    2. It provides stronger signal: positives must be clearly better than negatives
    3. It's more stable: margin prevents the model from just barely getting positives into top-K
    
    Key insight: For P@1, we need the best positive to be clearly better than the best negative.
    This loss ensures that if positives are in top-K, they have a margin over all negatives.
    """

    def __init__(self, k=3, margin=0.2, temperature=0.1, pos_weight=0.3):
        """
        Args:
            k: Target top-K positions
            margin: Minimum margin between positives in top-K and all negatives
            temperature: Temperature for softmax ranking
            pos_weight: Weight for maximizing positive similarities
        """
        super().__init__()
        self.k = k
        self.margin = margin
        self.temperature = temperature
        self.pos_weight = pos_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Get best positive similarity
        best_pos_sim, _ = torch.max(pos_sim, dim=1)  # (batch_size,)
        
        # Get best negative similarity
        best_neg_sim, _ = torch.max(neg_sim, dim=1)  # (batch_size,)
        
        # Loss 1: Ensure best positive is better than best negative with margin
        # This directly optimizes for P@1
        margin_loss = F.relu(self.margin - (best_pos_sim - best_neg_sim))
        
        # Loss 2: Ensure positives that are likely in top-K have margin over all negatives
        # Use softmax to get ranking probabilities
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)  # (batch_size, num_pos + num_neg)
        
        # Sort by probability to find top-K
        probs_sorted, indices_sorted = torch.sort(probs, dim=1, descending=True)
        top_k_probs = probs_sorted[:, :self.k]  # (batch_size, k)
        top_k_indices = indices_sorted[:, :self.k]  # (batch_size, k)
        
        # For each positive that appears in top-K, ensure it has margin over all negatives
        top_k_margin_loss = torch.zeros(batch_size, device=anchor.device)
        for i in range(batch_size):
            # Find positives in top-K
            pos_in_topk = top_k_indices[i] < num_pos  # (k,)
            if pos_in_topk.any():
                # Get similarities of positives in top-K
                pos_indices = top_k_indices[i][pos_in_topk]  # Indices of positives in top-K
                pos_sims_in_topk = pos_sim[i][pos_indices]  # Similarities of positives in top-K
                
                # Ensure these positives have margin over best negative
                best_neg = best_neg_sim[i]
                for pos_sim_val in pos_sims_in_topk:
                    top_k_margin_loss[i] += F.relu(self.margin - (pos_sim_val - best_neg))
        
        # Combine losses
        total_loss = margin_loss.mean() + top_k_margin_loss.mean()
        
        # Also maximize positive similarities
        if self.pos_weight > 0:
            avg_pos_sim = pos_sim.mean()
            pos_loss = self.pos_weight * (1.0 - avg_pos_sim)
            total_loss = total_loss + pos_loss
        
        return total_loss


class SoftTopKLoss(nn.Module):
    """Soft Top-K Loss: Uses smooth weighting that heavily emphasizes top positions.
    
    This is better than hard Hit@K because:
    1. It provides gradients even to items just outside top-K
    2. It uses exponential decay for position weights (like NDCG)
    3. It's more stable and provides better optimization signal
    
    Key insight: Instead of hard top-K, use soft position weights that decay exponentially.
    This way, items at rank 1 get maximum weight, rank 2 gets less, etc.
    """

    def __init__(self, k=3, temperature=0.1, decay_factor=2.0, pos_weight=0.3):
        """
        Args:
            k: Focus on top-K positions (but use soft weighting)
            temperature: Temperature for softmax
            decay_factor: How quickly position weights decay (higher = more focus on top)
            pos_weight: Weight for maximizing positive similarities
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.decay_factor = decay_factor
        self.pos_weight = pos_weight

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positives: (batch_size, num_positives, embed_dim)
            negatives: (batch_size, num_negatives, embed_dim)
        """
        batch_size = anchor.size(0)
        num_pos = positives.size(1)
        num_neg = negatives.size(1)

        # Compute similarities
        anchor_expanded = anchor.unsqueeze(1)
        pos_sim = F.cosine_similarity(anchor_expanded, positives, dim=2)  # (batch_size, num_pos)
        neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2)  # (batch_size, num_neg)

        # Combine all similarities
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch_size, num_pos + num_neg)
        
        # Get ranking probabilities
        scores = all_sim / self.temperature
        probs = F.softmax(scores, dim=1)  # (batch_size, num_pos + num_neg)
        
        # Compute position weights: exponential decay
        # Weight for position i = exp(-decay_factor * (i-1) / k)
        # This gives maximum weight to position 1, then decays
        num_items = num_pos + num_neg
        positions = torch.arange(1, num_items + 1, device=anchor.device, dtype=torch.float32)
        # Normalize positions by k, then apply exponential decay
        normalized_positions = (positions - 1.0) / self.k
        position_weights = torch.exp(-self.decay_factor * normalized_positions)  # (num_items,)
        
        # Sort by probability to get ranks
        probs_sorted, indices_sorted = torch.sort(probs, dim=1, descending=True)
        
        # For each positive, compute weighted probability based on its rank
        pos_weighted_probs = torch.zeros(batch_size, device=anchor.device)
        for i in range(batch_size):
            for j in range(num_pos):
                # Find rank of positive j
                rank = (indices_sorted[i] == j).nonzero(as_tuple=True)[0]
                if rank.numel() > 0:
                    rank_idx = rank[0].item()
                    # Weight by position and probability
                    pos_weighted_probs[i] += probs[i, j] * position_weights[rank_idx]
        
        # Loss = negative weighted probability (we want to maximize it)
        loss = -pos_weighted_probs.mean()
        
        # Also maximize positive similarities
        if self.pos_weight > 0:
            avg_pos_sim = pos_sim.mean()
            pos_loss = self.pos_weight * (1.0 - avg_pos_sim)
            loss = loss + pos_loss
        
        return loss


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, embeddings, clique_ids, device, batch_size=1000, use_metric_learning=False):
    """Compute Precision@1 and MAP@5 with refined embeddings or learned metric."""
    model.eval()

    embeddings_tensor = torch.FloatTensor(embeddings).to(device)

    # Transform embeddings through model
    refined = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            if use_metric_learning:
                refined.append(model._transform(batch).cpu().numpy())
            else:
                refined.append(model(batch).cpu().numpy())
    refined = np.vstack(refined)

    # Compute metrics using cosine distances on transformed embeddings
    n = len(refined)
    hits_at_1 = 0
    hits_at_5 = 0
    queries = 0
    total_processed = 0
    ap_scores = []

    for batch_start in tqdm(range(0, n, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, n)
        batch = refined[batch_start:batch_end]
        distances = cosine_distances(batch, refined)

        for i, row in enumerate(distances):
            query_idx = batch_start + i
            query_clique = clique_ids[query_idx]
            total_processed += 1

            n_covers = sum(1 for c in clique_ids if c == query_clique) - 1
            if n_covers == 0:
                continue

            queries += 1

            sorted_indices = np.argsort(row)
            sorted_indices = [idx for idx in sorted_indices if idx != query_idx][:100]

            if clique_ids[sorted_indices[0]] == query_clique:
                hits_at_1 += 1

            top5_cliques = [clique_ids[idx] for idx in sorted_indices[:5]]
            if query_clique in top5_cliques:
                hits_at_5 += 1

            hits = 0
            ap = 0
            for k, idx in enumerate(sorted_indices[:5]):
                if clique_ids[idx] == query_clique:
                    hits += 1
                    ap += hits / (k + 1)
            denominator = max(1, min(5, n_covers))
            ap_scores.append(ap / denominator)

    # Count how many songs were skipped (single-song cliques)
    skipped = total_processed - queries
    total_songs = total_processed  # Use total_processed as total_songs for percentage calculation
    
    # Warn if very few queries (unreliable metrics)
    if queries < 10:
        print(f"\n⚠️  WARNING: Only {queries} queries evaluated! Metrics may be unreliable.")
        print(f"   This usually means most songs in the set have no covers (single-song cliques).")
        print(f"   Skipped {skipped} songs (single-song cliques, no covers to find)")
    
    if queries == 0:
        print(f"\n❌ ERROR: No queries could be evaluated!")
        print(f"   All songs in this set are single-song cliques (no covers to find).")
        print(f"   Skipped {skipped} songs")
        return {
            'precision_at_1': 0.0,
            'hit_at_5': 0.0,
            'map_at_5': 0.0,
            'num_queries': 0
        }
    
    # Log evaluation stats
    if skipped > 0 and total_songs > 0:
        skip_pct = (skipped / total_songs) * 100
        print(f"  Evaluation: {queries} queries evaluated, {skipped} skipped ({skip_pct:.1f}% single-song cliques)")
    
    return {
        'precision_at_1': hits_at_1 / queries if queries > 0 else 0.0,
        'hit_at_5': hits_at_5 / queries if queries > 0 else 0.0,
        'map_at_5': np.mean(ap_scores) if ap_scores else 0.0,
        'num_queries': queries
    }


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading embeddings...")
    embeddings = np.loadtxt(args.embeddings, delimiter='\t')
    print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    print("Loading metadata...")
    metadata = pd.read_csv(args.metadata, sep='\t')

    print("Loading ground truth...")
    ground_truth = pd.read_csv(args.ground_truth)
    ground_truth['youtube_id'] = ground_truth['youtube_url'].str.extract(r'v=([A-Za-z0-9_-]{11})')
    id_to_clique = dict(zip(ground_truth['youtube_id'], ground_truth['clique']))

    # Map cliques
    clique_ids = [id_to_clique.get(vid, f"unknown_{i}") for i, vid in enumerate(metadata['youtube_id'])]

    # Filter to songs with known cliques
    valid_mask = [not str(c).startswith('unknown_') for c in clique_ids]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]

    embeddings_valid = embeddings[valid_indices]
    clique_ids_valid = [clique_ids[i] for i in valid_indices]

    print(f"Songs with ground truth: {len(embeddings_valid)}")

    # Split into train and validation sets
    if args.val_split > 0:
        # Always use clique-aware split to keep covers together
        # Stratification by clique_id doesn't keep songs from same clique together!
        # We need to split by clique (keep all songs from a clique together)
        from collections import defaultdict
        clique_to_indices = defaultdict(list)
        for idx, clique in enumerate(clique_ids_valid):
            clique_to_indices[clique].append(idx)
        
        # Get list of unique cliques
        unique_cliques = list(clique_to_indices.keys())
        
        # Try to split cliques (not individual songs)
        try:
            clique_train, clique_val = train_test_split(
                unique_cliques,
                test_size=args.val_split,
                random_state=42
            )
            # Build train/val indices from clique assignments
            train_indices = []
            val_indices = []
            for clique in clique_train:
                train_indices.extend(clique_to_indices[clique])
            for clique in clique_val:
                val_indices.extend(clique_to_indices[clique])
            print("Using clique-aware split (all songs from a clique stay together)")
        except ValueError:
            # If stratification fails (too many unique cliques), use balanced split
            print("Warning: Stratification failed (too many unique cliques)")
            print("Using balanced split to ensure similar difficulty...")
            
            # Group by clique size to balance difficulty
            from collections import Counter
            clique_counts = Counter(clique_ids_valid)
            
            # Separate single-song cliques (can't be evaluated) from multi-song cliques
            single_cliques = [c for c, count in clique_counts.items() if count == 1]
            multi_cliques = [c for c, count in clique_counts.items() if count >= 2]
            
            # Create groups: small cliques (2 songs), medium (3-5), large (6+)
            small_cliques = [c for c in multi_cliques if clique_counts[c] == 2]
            medium_cliques = [c for c in multi_cliques if 3 <= clique_counts[c] <= 5]
            large_cliques = [c for c in multi_cliques if clique_counts[c] >= 6]
            
            print(f"  Clique distribution: {len(single_cliques)} single (skipped), {len(small_cliques)} small, {len(medium_cliques)} medium, {len(large_cliques)} large")
            
            # Split each multi-song group separately to maintain balance
            train_indices = []
            val_indices = []
            
            for clique_group in [small_cliques, medium_cliques, large_cliques]:
                if not clique_group:
                    continue
                # Get indices for this group
                group_indices = [i for i, c in enumerate(clique_ids_valid) if c in clique_group]
                if len(group_indices) < 2:
                    continue
                
                # Split this group - ensure we keep cliques together
                # Group indices by clique
                clique_to_indices = {}
                for idx in group_indices:
                    clique = clique_ids_valid[idx]
                    if clique not in clique_to_indices:
                        clique_to_indices[clique] = []
                    clique_to_indices[clique].append(idx)
                
                # Split cliques (not individual songs) to keep covers together
                clique_list = list(clique_to_indices.keys())
                try:
                    clique_train, clique_val = train_test_split(
                        clique_list,
                        test_size=args.val_split,
                        random_state=42
                    )
                    # Add all songs from train cliques to train, val cliques to val
                    for clique in clique_train:
                        train_indices.extend(clique_to_indices[clique])
                    for clique in clique_val:
                        val_indices.extend(clique_to_indices[clique])
                except:
                    # If split fails, add all to train
                    for clique in clique_list:
                        train_indices.extend(clique_to_indices[clique])
            
            # Add single-song cliques mostly to train (they can't be evaluated anyway)
            single_indices = [i for i, c in enumerate(clique_ids_valid) if c in single_cliques]
            if single_indices:
                # Put 90% in train, 10% in val (they won't be evaluated but keep dataset balanced)
                n_single_val = int(len(single_indices) * 0.1)
                train_indices.extend(single_indices[n_single_val:])
                val_indices.extend(single_indices[:n_single_val])
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        embeddings_train = embeddings_valid[train_indices]
        clique_ids_train = [clique_ids_valid[i] for i in train_indices]
        embeddings_val = embeddings_valid[val_indices]
        clique_ids_val = [clique_ids_valid[i] for i in val_indices]
        
        print(f"Train: {len(embeddings_train)}, Val: {len(embeddings_val)}")
        
        # Analyze split balance
        from collections import Counter
        train_clique_counts = Counter(clique_ids_train)
        val_clique_counts = Counter(clique_ids_val)
        
        train_single = sum(1 for c, count in train_clique_counts.items() if count == 1)
        train_multi = sum(1 for c, count in train_clique_counts.items() if count > 1)
        val_single = sum(1 for c, count in val_clique_counts.items() if count == 1)
        val_multi = sum(1 for c, count in val_clique_counts.items() if count > 1)
        
        # Count evaluable songs (songs in multi-song cliques)
        train_evaluable = sum(count for c, count in train_clique_counts.items() if count > 1)
        val_evaluable = sum(count for c, count in val_clique_counts.items() if count > 1)
        
        print(f"  Train: {train_multi} multi-song cliques ({train_evaluable} evaluable songs), {train_single} single-song cliques")
        print(f"  Val:   {val_multi} multi-song cliques ({val_evaluable} evaluable songs), {val_single} single-song cliques")
        
        # Check if cliques are being split (bad!) - this causes low precision@1
        train_clique_set = set(c for c, count in train_clique_counts.items() if count > 1)
        val_clique_set = set(c for c, count in val_clique_counts.items() if count > 1)
        split_cliques = train_clique_set & val_clique_set
        if split_cliques:
            print(f"\n⚠️  CRITICAL: {len(split_cliques)} cliques are split across train/val!")
            print(f"   This means covers are being separated, causing very low precision@1.")
            print(f"   Example: If clique A has songs [1,2] and song 1 is in train, song 2 in val,")
            print(f"   then when evaluating train, song 1 can't find its cover (song 2 is in val).")
            print(f"   ")
            print(f"   The split should keep all songs from a clique together.")
            print(f"   Consider using --val-split 0.0 to train on full dataset.")
        
        # Warn if validation set has too few evaluable songs
        if val_evaluable < 100:
            print(f"\n⚠️  WARNING: Validation set has only {val_evaluable} evaluable songs!")
            print(f"   This may lead to unreliable metrics.")
            print(f"   Consider:")
            print(f"   - Using --val-split 0.0 to train on full dataset")
            print(f"   - Or reducing validation split size (--val-split 0.1)")
    else:
        embeddings_train = embeddings_valid
        clique_ids_train = clique_ids_valid
        embeddings_val = None
        clique_ids_val = None

    # Evaluate baseline - ALWAYS use cosine distance (not learned metric)
    print("\n=== Baseline (raw embeddings) ===")
    baseline_model = nn.Identity().to(device)
    
    # Always evaluate on train set first (this is the "true" baseline)
    train_baseline = evaluate(baseline_model, embeddings_train, clique_ids_train, device, use_metric_learning=False)
    print(f"Train set baseline:")
    print(f"  Precision@1: {train_baseline['precision_at_1']:.4f} ({train_baseline['precision_at_1']*100:.1f}%)")
    print(f"  Hit@5:       {train_baseline['hit_at_5']:.4f} ({train_baseline['hit_at_5']*100:.1f}%)")
    print(f"  MAP@5:       {train_baseline['map_at_5']:.4f} ({train_baseline['map_at_5']*100:.1f}%)")
    print(f"  Queries:     {train_baseline['num_queries']}")
    
    # Evaluate on validation set if available
    if embeddings_val is not None:
        val_baseline = evaluate(baseline_model, embeddings_val, clique_ids_val, device, use_metric_learning=False)
        print(f"Validation set baseline:")
        print(f"  Precision@1: {val_baseline['precision_at_1']:.4f} ({val_baseline['precision_at_1']*100:.1f}%)")
        print(f"  Hit@5:       {val_baseline['hit_at_5']:.4f} ({val_baseline['hit_at_5']*100:.1f}%)")
        print(f"  MAP@5:       {val_baseline['map_at_5']:.4f} ({val_baseline['map_at_5']*100:.1f}%)")
        print(f"  Queries:     {val_baseline['num_queries']}")
        
        # Check for significant imbalance
        p1_diff = abs(train_baseline['precision_at_1'] - val_baseline['precision_at_1'])
        if p1_diff > 0.15:  # More than 15% difference
            print(f"\n⚠️  WARNING: Large baseline difference detected ({p1_diff*100:.1f}%)")
            print(f"   This suggests the train/val split may be imbalanced.")
            print(f"   Possible causes:")
            print(f"   1. Validation set has harder cases (fewer covers per song)")
            print(f"   2. Validation set has different musical styles/genres")
            print(f"   3. Random split created an uneven distribution")
            print(f"   ")
            print(f"   Recommendations:")
            print(f"   - The model will be optimized for validation set performance")
            print(f"   - Consider using --val-split 0.0 to train on full dataset")
            print(f"   - Or use a different random_state for a more balanced split")
            print(f"   - Final evaluation will show improvement vs both baselines")
        
        baseline_metrics = val_baseline  # Use validation baseline for model selection
        eval_embeddings = embeddings_val
        eval_clique_ids = clique_ids_val
    else:
        baseline_metrics = train_baseline
        eval_embeddings = embeddings_train
        eval_clique_ids = clique_ids_train

    # Create model
    use_metric_learning = args.metric_learning
    if use_metric_learning:
        model = MetricLearner(
            input_dim=embeddings.shape[1],
            metric_type=args.metric_type,
            rank=args.metric_rank
        ).to(device)
        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        print(f"\nUsing Metric Learning ({args.metric_type})")
        # Debug: check if parameters require gradients
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Parameters requiring gradients: {params_with_grad}/{sum(1 for _ in model.parameters())}")
    else:
        model = EmbeddingRefiner(
            input_dim=embeddings.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            use_linear=args.linear
        ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataset and dataloader
    if args.loss in ['ranking', 'approx_ndcg', 'precision_at_1', 'hit_at_k', 'reciprocal_rank', 'top_k_margin', 'soft_top_k', 'multi_neg']:
        # multi_neg also needs multiple positives and negatives, so use RankingDataset
        dataset = RankingDataset(embeddings_train, clique_ids_train,
                                num_samples=args.triplets_per_epoch,
                                num_positives=args.num_positives,
                                num_negatives=args.num_negatives,
                                use_semi_hard=True)
    else:
        dataset = TripletDataset(embeddings_train, clique_ids_train,
                                 num_triplets=args.triplets_per_epoch,
                                 hard_negatives=args.hard_negatives)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Loss and optimizer
    if args.loss == 'triplet':
        criterion = TripletLoss(margin=args.margin)
    elif args.loss == 'multi_neg':
        criterion = MultiNegativeContrastiveLoss(temperature=args.temperature)
    elif args.loss == 'inbatch':
        criterion = InBatchNTXentLoss(temperature=args.temperature)
    elif args.loss == 'ranking':
        criterion = RankingLoss(margin=args.margin, top_k_weight=args.top_k_weight)
    elif args.loss == 'approx_ndcg':
        criterion = ApproxNDCGLoss(temperature=args.temperature, top_k=args.top_k)
    elif args.loss == 'precision_at_1':
        criterion = PrecisionAt1Loss(temperature=args.temperature)
    elif args.loss == 'direct_pat1':
        # Use smaller margin for direct_pat1 (it's more sensitive)
        pat1_margin = min(args.margin, 0.15)  # Cap at 0.15 for stability
        criterion = DirectPAt1Loss(temperature=args.temperature, margin=pat1_margin, hard_negative_weight=args.top_k_weight)
    elif args.loss == 'listnet':
        criterion = ListNetLoss(temperature=args.temperature)
    elif args.loss == 'hit_at_k':
        criterion = HitAtKLoss(k=args.hit_k, temperature=args.temperature, pos_weight=args.pos_weight)
    elif args.loss == 'reciprocal_rank':
        criterion = ReciprocalRankLoss(temperature=args.temperature, pos_weight=args.pos_weight)
    elif args.loss == 'top_k_margin':
        criterion = TopKMarginLoss(k=args.hit_k, margin=args.margin, temperature=args.temperature, pos_weight=args.pos_weight)
    elif args.loss == 'soft_top_k':
        criterion = SoftTopKLoss(k=args.hit_k, temperature=args.temperature, decay_factor=args.decay_factor, pos_weight=args.pos_weight)
    else:  # ntxent
        criterion = NTXentLoss(temperature=args.temperature)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_precision = baseline_metrics['precision_at_1']
    # Initialize with current model state (baseline) as fallback
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    best_val_precision = baseline_metrics['precision_at_1'] if embeddings_val is not None else None
    prev_loss = float('inf')

    print(f"\n=== Training ===")
    print(f"Initial learning rate: {args.lr}")
    print(f"Optimizer: Adam with weight decay {args.weight_decay}")
    print(f"Loss function: {args.loss}")
    if args.loss in ['ranking', 'approx_ndcg', 'precision_at_1', 'direct_pat1', 'listnet']:
        print(f"  - Positives per sample: {args.num_positives}")
        print(f"  - Negatives per sample: {args.num_negatives}")
    
    # Warning if using complex loss with high learning rate
    if args.loss in ['direct_pat1', 'listnet', 'approx_ndcg'] and args.lr > 0.002:
        print(f"\n⚠️  Warning: Using complex loss '{args.loss}' with high learning rate {args.lr}")
        print(f"   Consider reducing learning rate (--lr 0.0005) or using simpler loss (--loss triplet)")
    for epoch in range(args.epochs):
        # Recompute hard negatives with current model's embeddings each epoch
        if epoch > 0 and hasattr(dataset, 'update_embeddings'):
            model.eval()
            with torch.no_grad():
                train_tensor = torch.FloatTensor(embeddings_train).to(device)
                transformed = []
                for i in range(0, len(embeddings_train), 1000):
                    batch = train_tensor[i:i+1000]
                    if use_metric_learning:
                        transformed.append(model._transform(batch).cpu().numpy())
                    else:
                        transformed.append(model(batch).cpu().numpy())
                transformed = np.vstack(transformed)
            dataset.update_embeddings(transformed)

        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_data in pbar:
            # Handle different dataset formats
            if use_metric_learning:
                if args.loss in ['ranking', 'approx_ndcg', 'precision_at_1', 'direct_pat1', 'listnet', 'hit_at_k', 'reciprocal_rank', 'top_k_margin', 'soft_top_k', 'multi_neg']:
                    anchor, positives, negatives = batch_data
                    anchor = anchor.to(device)
                    positives = positives.to(device)
                    negatives = negatives.to(device)

                    batch_size = anchor.size(0)
                    num_pos = positives.size(1)
                    num_neg = negatives.size(1)
                    embed_dim = anchor.size(-1)

                    # Transform all embeddings through learned metric
                    anchor_out = model._transform(anchor)  # (B, D)
                    positives_out = model._transform(positives.reshape(-1, embed_dim)).view(batch_size, num_pos, -1)
                    negatives_out = model._transform(negatives.reshape(-1, embed_dim)).view(batch_size, num_neg, -1)

                    # For multi_neg loss: select best positive (highest similarity to anchor)
                    if args.loss == 'multi_neg':
                        pos_sims = F.cosine_similarity(anchor_out.unsqueeze(1), positives_out, dim=2)
                        best_pos_idx = pos_sims.argmax(dim=1)  # (B,)
                        best_positive = positives_out[torch.arange(batch_size), best_pos_idx]  # (B, D)
                        loss = criterion(anchor_out, best_positive, negatives_out)
                    else:
                        loss = criterion(anchor_out, positives_out, negatives_out)
                else:
                    # Triplet loss with metric learning
                    anchor, positive, negative = batch_data
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)

                    pos_dist = model.compute_distance(anchor, positive)
                    neg_dist = model.compute_distance(anchor, negative)
                    loss = F.relu(pos_dist - neg_dist + args.margin).mean()
            else:
                # Original embedding refinement approach
                if args.loss in ['ranking', 'approx_ndcg', 'precision_at_1', 'direct_pat1', 'listnet']:
                    anchor, positives, negatives = batch_data
                    anchor = anchor.to(device)
                    positives = positives.to(device)
                    negatives = negatives.to(device)
                    
                    anchor_out = model(anchor)
                    positives_out = model(positives.view(-1, positives.size(-1))).view(
                        positives.size(0), positives.size(1), -1)
                    negatives_out = model(negatives.view(-1, negatives.size(-1))).view(
                        negatives.size(0), negatives.size(1), -1)
                    
                    loss = criterion(anchor_out, positives_out, negatives_out)
                else:
                    anchor, positive, negative = batch_data
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)

                    # Forward
                    anchor_out = model(anchor)
                    positive_out = model(positive)

                    # Handle multiple negatives (3D tensor) vs single negative (2D tensor)
                    if args.loss == 'multi_neg':
                        # negative: (batch_size, num_negatives, embed_dim)
                        batch_size, num_neg, embed_dim = negative.shape
                        negative_flat = negative.view(-1, embed_dim)
                        negative_out_flat = model(negative_flat)
                        negative_out = negative_out_flat.view(batch_size, num_neg, -1)
                    else:
                        negative_out = model(negative)

                    loss = criterion(anchor_out, positive_out, negative_out)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        # Evaluate on validation set if available, otherwise on train set
        if embeddings_val is not None:
            val_metrics = evaluate(model, embeddings_val, clique_ids_val, device, use_metric_learning=use_metric_learning)
            metrics = val_metrics
            eval_set = "Val"
        else:
            metrics = evaluate(model, embeddings_train, clique_ids_train, device, use_metric_learning=use_metric_learning)
            eval_set = "Train"
        
        # Check for embedding collapse and embedding spread
        if epoch % 2 == 0:  # Check every 2 epochs to avoid slowdown
            model.eval()
            with torch.no_grad():
                sample_emb = torch.FloatTensor(embeddings_train[:100]).to(device)
                if use_metric_learning:
                    refined_sample = model._transform(sample_emb)
                else:
                    refined_sample = model(sample_emb)
                # Compute average pairwise similarity
                pairwise_sim = F.cosine_similarity(
                    refined_sample.unsqueeze(1), 
                    refined_sample.unsqueeze(0), 
                    dim=2
                )
                # Exclude diagonal (self-similarity = 1.0)
                mask = ~torch.eye(pairwise_sim.size(0), dtype=torch.bool, device=pairwise_sim.device)
                avg_sim = pairwise_sim[mask].mean().item()
                std_sim = pairwise_sim[mask].std().item()
                
                if avg_sim > 0.95:
                    print(f"  ⚠️  Warning: Embeddings may be collapsing (avg similarity: {avg_sim:.3f}, std: {std_sim:.3f})")
                elif avg_sim < 0.1:
                    print(f"  ⚠️  Warning: Embeddings may be too spread out (avg similarity: {avg_sim:.3f}, std: {std_sim:.3f})")
                else:
                    print(f"  Embedding stats: avg_sim={avg_sim:.3f}, std={std_sim:.3f}")
            model.train()

        # Check if loss is decreasing
        if epoch == 0:
            loss_trend = "→"
        else:
            loss_trend = "↓" if avg_loss < prev_loss else "↑"
        
        print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f} {loss_trend}, {eval_set} P@1={metrics['precision_at_1']*100:.1f}%, "
              f"Hit@5={metrics['hit_at_5']*100:.1f}%, MAP@5={metrics['map_at_5']*100:.1f}%")
        
        prev_loss = avg_loss

        # Save best model based on validation set if available
        if embeddings_val is not None:
            if metrics['precision_at_1'] > best_val_precision:
                best_val_precision = metrics['precision_at_1']
                best_precision = metrics['precision_at_1']
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"  New best! Val P@1 improved to {best_val_precision*100:.1f}%")
            else:
                patience_counter += 1
                if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                    print(f"  Early stopping after {patience_counter} epochs without improvement")
                    break
        else:
            # No validation set, use train metrics
            if metrics['precision_at_1'] > best_precision:
                best_precision = metrics['precision_at_1']
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best! P@1 improved to {best_precision*100:.1f}%")
        
        # Safety check: if model is performing worse than baseline, revert
        if metrics['precision_at_1'] < baseline_metrics['precision_at_1'] * 0.9:  # 10% worse
            print(f"  ⚠️  Warning: Model performance ({metrics['precision_at_1']*100:.1f}%) is significantly worse than baseline ({baseline_metrics['precision_at_1']*100:.1f}%)")
            print(f"  This may indicate overfitting or training instability.")
            patience_counter += 1  # Count this as a bad epoch
            # Early stop if performance degrades too much
            if metrics['precision_at_1'] < baseline_metrics['precision_at_1'] * 0.5:  # 50% worse
                print(f"  🛑 CRITICAL: Performance dropped below 50% of baseline. Stopping training.")
                print(f"  The model is learning incorrectly. Consider:")
                print(f"     - Using simpler loss (--loss triplet)")
                print(f"     - Lower learning rate (--lr 0.0005)")
                print(f"     - Linear model (--linear)")
                print(f"     - More regularization (--weight-decay 1e-3)")
                break

    # Load best model and save
    # best_model_state should always be set (initialized with baseline)
    model.load_state_dict(best_model_state)
    
    # Verify the best model is actually better than baseline before saving
    use_baseline = False
    if embeddings_val is not None:
        best_model_metrics = evaluate(model, embeddings_val, clique_ids_val, device, use_metric_learning=use_metric_learning)
        if best_model_metrics['precision_at_1'] < baseline_metrics['precision_at_1']:
            print(f"\n⚠️  Warning: Best model ({best_model_metrics['precision_at_1']*100:.1f}%) is worse than baseline ({baseline_metrics['precision_at_1']*100:.1f}%)")
            print(f"   Model did not improve. Using baseline (identity) model instead.")
            use_baseline = True
            # Revert to identity model (baseline)
            if use_metric_learning:
                # For metric learning, create a baseline MetricLearner that returns cosine distance
                # Revert to identity metric (baseline)
                model = MetricLearner(
                    input_dim=embeddings.shape[1],
                    metric_type=args.metric_type,
                    rank=args.metric_rank
                ).to(device)
                # W is already initialized near identity by __init__
                with torch.no_grad():
                    dim = embeddings.shape[1]
                    rank = args.metric_rank if args.metric_rank else dim
                    rank = min(rank, dim)
                    if rank == dim:
                        model.W.data = torch.eye(dim).to(device)
                    else:
                        model.W.data = torch.eye(rank, dim).to(device)
            else:
                # For embedding refinement, create identity EmbeddingRefiner
                model = EmbeddingRefiner(
                    input_dim=embeddings.shape[1],
                    hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim,
                    use_linear=True  # Use linear for identity-like behavior
                ).to(device)
                # Set linear layer to identity
                with torch.no_grad():
                    if embeddings.shape[1] == args.output_dim:
                        model.mlp.weight.copy_(torch.eye(embeddings.shape[1]))
                    else:
                        # Use projection that preserves as much as possible
                        nn.init.xavier_uniform_(model.mlp.weight)
    
    if not use_baseline:
        torch.save(best_model_state, args.output_model)
        print(f"\nSaved best model to {args.output_model}")
    else:
        print(f"\nNo model saved (using baseline)")

    # Generate refined embeddings for all songs
    print("\nGenerating refined embeddings for all songs...")
    model.eval()
    embeddings_tensor = torch.FloatTensor(embeddings).to(device)
    refined_all = []

    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), 1000)):
            batch = embeddings_tensor[i:i+1000]
            if use_metric_learning:
                refined_all.append(model._transform(batch).cpu().numpy())
            else:
                refined_all.append(model(batch).cpu().numpy())

    refined_all = np.vstack(refined_all)
    np.savetxt(args.output_embeddings, refined_all, delimiter='\t', fmt='%.6f')
    print(f"Saved refined embeddings to {args.output_embeddings}")

    # Final evaluation on the same set as baseline for fair comparison
    print("\n=== Final Results ===")
    # Model should already be loaded with best_model_state (or baseline if reverted)
    
    # Evaluate on the same set as baseline (validation if available, else train)
    final_metrics = evaluate(model, eval_embeddings, eval_clique_ids, device, use_metric_learning=use_metric_learning)
    final_precision = final_metrics['precision_at_1']
    
    # Show comparison - always show train baseline for reference
    if embeddings_val is not None:
        train_baseline_p1 = train_baseline['precision_at_1']*100
        val_baseline_p1 = baseline_metrics['precision_at_1']*100
        
        print(f"Train Baseline P@1:     {train_baseline_p1:.1f}%")
        print(f"Val Baseline P@1:       {val_baseline_p1:.1f}%")
        print(f"Refined P@1 (on Val):    {final_precision*100:.1f}%")
        
        # Improvement over validation baseline (what we optimized for)
        improvement_val = (final_precision - baseline_metrics['precision_at_1'])*100
        print(f"Improvement (vs Val):    {improvement_val:+.1f}%")
        
        # Also show improvement over train baseline for reference
        improvement_train = (final_precision - train_baseline['precision_at_1'])*100
        print(f"Improvement (vs Train):  {improvement_train:+.1f}%")
        
        if best_val_precision is not None:
            print(f"Best Val P@1:           {best_val_precision*100:.1f}%")
    else:
        # No validation split - simple comparison
        print(f"Baseline P@1:  {baseline_metrics['precision_at_1']*100:.1f}%")
        print(f"Refined P@1:   {final_precision*100:.1f}%")
        improvement = (final_precision - baseline_metrics['precision_at_1'])*100
        print(f"Improvement:   {improvement:+.1f}%")
    
    # Warning if performance degraded
    if final_precision < baseline_metrics['precision_at_1']:
        print(f"\n⚠️  WARNING: Model performance degraded!")
        print(f"   This suggests the embedding space may not be suitable for refinement,")
        print(f"   or there's a fundamental mismatch between training and evaluation.")
        print(f"\n   Possible causes:")
        print(f"   1. Hard negatives computed on raw embeddings become outdated during training")
        print(f"   2. Training negatives don't match the negatives that rank high in evaluation")
        print(f"   3. The model is learning a degenerate solution that minimizes loss but hurts P@1")
        print(f"\n   Recommendations:")
        print(f"   - The baseline embeddings (63.1% P@1) may already be near-optimal")
        print(f"   - Consider metric learning on the raw embeddings instead of refinement")
        print(f"   - Or use a different approach: fine-tune the original embedding model")
        print(f"   - If you must refine, try: --linear --lr 0.0001 --weight-decay 1e-3 --epochs 3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train embedding refiner for cover detection')

    # Data paths
    parser.add_argument('--embeddings', default='vectors_without_metadata_v5.tsv',
                       help='Path to embeddings TSV file')
    parser.add_argument('--metadata', default='metadata_clean_v5.tsv',
                       help='Path to metadata TSV file')
    parser.add_argument('--ground-truth', default='videos_to_test.csv',
                       help='Path to ground truth CSV with clique IDs')

    # Model architecture
    parser.add_argument('--metric-learning', action='store_true', default=True,
                       help='Use metric learning instead of embedding refinement (learns distance metric)')
    parser.add_argument('--no-metric-learning', action='store_false', dest='metric_learning',
                       help='Disable metric learning, use embedding refinement instead')
    parser.add_argument('--metric-type', choices=['mahalanobis', 'bilinear'], default='bilinear',
                       help='Type of metric to learn: mahalanobis (d=||W(x-y)||^2) or bilinear (sim=x^T M y)')
    parser.add_argument('--metric-rank', type=int, default=None,
                       help='Rank of metric matrix (None = full rank, lower = fewer parameters)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension of MLP')
    parser.add_argument('--output-dim', type=int, default=128,
                       help='Output embedding dimension')
    parser.add_argument('--linear', action='store_true',
                       help='Use simple linear projection instead of MLP')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (increased default to prevent overfitting)')
    parser.add_argument('--triplets-per-epoch', type=int, default=100000,
                       help='Number of triplets per epoch')
    parser.add_argument('--hard-negatives', action='store_true',
                       help='Use hard negative mining (closest non-cover)')

    # Loss
    parser.add_argument('--loss', 
                       choices=['triplet', 'ntxent', 'multi_neg', 'inbatch', 'ranking', 'approx_ndcg', 'precision_at_1', 'direct_pat1', 'listnet', 'hit_at_k', 'reciprocal_rank', 'top_k_margin', 'soft_top_k'], 
                       default='multi_neg',
                       help='Loss function: triplet (recommended to start), ntxent, multi_neg, inbatch, ranking, approx_ndcg, precision_at_1, direct_pat1, listnet, hit_at_k, reciprocal_rank (MRR, often best for P@1), top_k_margin (ensures margin in top-K), soft_top_k (smooth top-K weighting)')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Triplet/Ranking loss margin (lower for direct_pat1, higher for triplet)')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Contrastive loss temperature (lower = harder)')
    parser.add_argument('--num-hard-negatives', type=int, default=5,
                       help='Number of hard negatives per sample (for multi_neg loss)')
    parser.add_argument('--num-positives', type=int, default=2,
                       help='Number of positives per anchor (for ranking losses)')
    parser.add_argument('--num-negatives', type=int, default=50,
                       help='Number of negatives per anchor (for ranking losses)')
    parser.add_argument('--top-k-weight', type=float, default=2.0,
                       help='Extra weight for top-k confusing negatives (for ranking loss)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Top-k positions to focus on (for approx_ndcg loss)')
    parser.add_argument('--hit-k', type=int, default=3,
                       help='K for Hit@K/top_K_margin/soft_top_k losses (default: 3, optimizes for at least one positive in top-K)')
    parser.add_argument('--pos-weight', type=float, default=0.3,
                       help='Weight for maximizing positive similarity term (for ranking/hit_at_k/reciprocal_rank losses, 0 = disabled)')
    parser.add_argument('--decay-factor', type=float, default=2.0,
                       help='Decay factor for soft_top_k loss (higher = more focus on top positions, default: 2.0)')
    
    # Validation and early stopping
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (0 = no validation)')
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience (0 = disabled)')

    # Output
    parser.add_argument('--output-model', default='embedding_refiner.pt',
                       help='Path to save trained model')
    parser.add_argument('--output-embeddings', default='vectors_refined.tsv',
                       help='Path to save refined embeddings')

    args = parser.parse_args()
    train(args)
