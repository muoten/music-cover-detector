#!/usr/bin/env python3
"""Compute Precision@1 for CoverHunter embeddings."""

import json
import numpy as np
from pathlib import Path

def load_vectors(csv_path, expected_dim=128):
    """Load embeddings from vectors.csv format."""
    embeddings = {}
    with open(csv_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid = parts[0]
            # Extract video ID from path like /tmp/.../VIDEO_ID.wav
            if '/' in vid:
                vid = vid.split('/')[-1].replace('.wav', '')
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == expected_dim:
                    embeddings[vid] = vec
            except:
                continue
    return embeddings

def compute_p1(embeddings, cover_map):
    """Compute Precision@1."""
    # Build list of videos that have covers
    vids_with_covers = []
    for vid in embeddings:
        if vid in cover_map and len(cover_map[vid]) > 0:
            # Check if at least one cover is in our embeddings
            covers_in_dataset = [c for c in cover_map[vid] if c in embeddings]
            if covers_in_dataset:
                vids_with_covers.append(vid)

    print(f"Songs with embeddings: {len(embeddings)}")
    print(f"Songs with verified covers in dataset: {len(vids_with_covers)}")

    # Build embedding matrix for fast similarity computation
    vid_list = list(embeddings.keys())
    vid_to_idx = {v: i for i, v in enumerate(vid_list)}

    emb_matrix = np.array([embeddings[v] for v in vid_list])
    # L2 normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-8)

    # Compute P@1
    hits = 0
    total = 0

    for vid in vids_with_covers:
        idx = vid_to_idx[vid]
        query_emb = emb_matrix[idx:idx+1]

        # Compute similarities to all songs
        similarities = (emb_matrix @ query_emb.T).flatten()

        # Set self-similarity to -inf
        similarities[idx] = -np.inf

        # Find nearest neighbor
        nn_idx = np.argmax(similarities)
        nn_vid = vid_list[nn_idx]

        # Check if nearest neighbor is a true cover
        true_covers = set(cover_map[vid])
        if nn_vid in true_covers:
            hits += 1
        total += 1

    p1 = hits / total if total > 0 else 0
    print(f"Hits: {hits}, Total: {total}")
    print(f"Precision@1: {p1:.4f} ({p1*100:.1f}%)")
    return p1

def main():
    base_path = Path(__file__).parent

    # CoverHunter
    print("=" * 50)
    print("COVERHUNTER (128-dim, YouTube full audio)")
    print("=" * 50)
    print("Loading CoverHunter embeddings...")
    embeddings_ch = load_vectors(base_path / "vectors.csv", expected_dim=128)

    print("Loading verified covers...")
    with open(base_path / "docs/verified_covers.json", 'r') as f:
        cover_map_ch = json.load(f)

    print("\nComputing Precision@1 for CoverHunter...")
    p1_ch = compute_p1(embeddings_ch, cover_map_ch)

    # VINet
    print("\n" + "=" * 50)
    print("VINET (512-dim, iTunes 30s previews)")
    print("=" * 50)
    print("Loading VINet embeddings...")
    embeddings_vinet = load_vectors(base_path / "vectors_vinet.csv", expected_dim=512)

    print("Loading verified covers...")
    with open(base_path / "docs/verified_covers_vinet.json", 'r') as f:
        cover_map_vinet = json.load(f)

    print("\nComputing Precision@1 for VINet...")
    p1_vinet = compute_p1(embeddings_vinet, cover_map_vinet)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"CoverHunter P@1: {p1_ch:.4f} ({p1_ch*100:.1f}%)")
    print(f"VINet P@1:       {p1_vinet:.4f} ({p1_vinet*100:.1f}%)")

    return p1_ch, p1_vinet

if __name__ == "__main__":
    main()
