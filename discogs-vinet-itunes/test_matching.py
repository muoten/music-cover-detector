#!/usr/bin/env python3
"""
Test iTunes matching strategies on collision pairs.
Goal: Find better ways to avoid collisions (different YouTube songs matching same iTunes preview).
"""

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline import (
    get_youtube_title,
    parse_artist_track,
    STRIP_PATTERNS,
)
import re


def compute_hash(embedding):
    emb_bytes = embedding.astype(np.float32).tobytes()
    return hashlib.md5(emb_bytes).hexdigest()[:8]


def load_vectors(csv_path):
    embeddings = {}
    with open(csv_path, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid = parts[0]
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == 512:
                    embeddings[vid] = vec
            except:
                continue
    return embeddings


def find_collision_pairs(embeddings, limit=20):
    """Find pairs of songs with same hash."""
    hash_to_vids = {}
    for vid, emb in embeddings.items():
        h = compute_hash(emb)
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    pairs = []
    for h, vids in hash_to_vids.items():
        if len(vids) >= 2:
            pairs.append((vids[0], vids[1], h))
            if len(pairs) >= limit:
                break
    return pairs


def search_itunes_detailed(query, limit=5):
    """Search iTunes and return detailed results."""
    params = urllib.parse.urlencode({
        'term': query,
        'media': 'music',
        'entity': 'song',
        'limit': str(limit),
    })
    url = f"https://itunes.apple.com/search?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('results', [])
    except Exception as e:
        return []


def analyze_collision_pair(vid1, vid2):
    """Analyze why two videos might have matched the same iTunes preview."""
    print(f"\n{'='*60}")
    print(f"Collision pair: {vid1} vs {vid2}")
    print('='*60)

    # Get YouTube titles
    title1 = get_youtube_title(vid1)
    title2 = get_youtube_title(vid2)
    print(f"\nYouTube titles:")
    print(f"  1: {title1}")
    print(f"  2: {title2}")

    # Parse artist/track
    artist1, track1 = parse_artist_track(title1)
    artist2, track2 = parse_artist_track(title2)
    print(f"\nParsed (current method):")
    print(f"  1: artist='{artist1}', track='{track1}'")
    print(f"  2: artist='{artist2}', track='{track2}'")

    # Build search queries
    query1 = f"{artist1} {track1}" if artist1 else track1
    query2 = f"{artist2} {track2}" if artist2 else track2
    print(f"\nSearch queries:")
    print(f"  1: '{query1}'")
    print(f"  2: '{query2}'")

    # Search iTunes
    time.sleep(0.3)
    results1 = search_itunes_detailed(query1)
    time.sleep(0.3)
    results2 = search_itunes_detailed(query2)

    print(f"\niTunes results for query 1:")
    for i, r in enumerate(results1[:3]):
        print(f"  {i+1}. {r.get('artistName')} - {r.get('trackName')}")
        print(f"     Preview: {r.get('previewUrl', 'N/A')[:60]}...")

    print(f"\niTunes results for query 2:")
    for i, r in enumerate(results2[:3]):
        print(f"  {i+1}. {r.get('artistName')} - {r.get('trackName')}")
        print(f"     Preview: {r.get('previewUrl', 'N/A')[:60]}...")

    # Check if top results are the same
    if results1 and results2:
        preview1 = results1[0].get('previewUrl', '')
        preview2 = results2[0].get('previewUrl', '')
        if preview1 == preview2:
            print(f"\n⚠️  SAME PREVIEW URL - Both matched to: {results1[0].get('artistName')} - {results1[0].get('trackName')}")
        else:
            print(f"\n✓ Different preview URLs")

    return {
        'vid1': vid1, 'vid2': vid2,
        'title1': title1, 'title2': title2,
        'artist1': artist1, 'track1': track1,
        'artist2': artist2, 'track2': track2,
        'query1': query1, 'query2': query2,
        'results1': results1[:3] if results1 else [],
        'results2': results2[:3] if results2 else [],
    }


def suggest_improvements(analyses):
    """Analyze patterns and suggest improvements."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    same_preview = 0
    same_query = 0
    both_no_artist = 0

    for a in analyses:
        if a['results1'] and a['results2']:
            p1 = a['results1'][0].get('previewUrl', '')
            p2 = a['results2'][0].get('previewUrl', '')
            if p1 == p2:
                same_preview += 1
        if a['query1'] == a['query2']:
            same_query += 1
        if not a['artist1'] and not a['artist2']:
            both_no_artist += 1

    print(f"\nOut of {len(analyses)} collision pairs:")
    print(f"  - {same_preview} have same iTunes preview URL")
    print(f"  - {same_query} have identical search queries")
    print(f"  - {both_no_artist} have no artist parsed for either song")

    print("\nPossible improvements:")
    print("  1. Add YouTube channel name as artist fallback")
    print("  2. Use more specific queries (add year, album)")
    print("  3. Compare iTunes results to YouTube title before accepting")
    print("  4. Try multiple search variations")
    print("  5. Use trackId to detect duplicates before embedding")


def main():
    vectors_path = os.path.join(SCRIPT_DIR, 'vectors_vinet.csv')

    print("Loading embeddings...")
    embeddings = load_vectors(vectors_path)
    print(f"Loaded {len(embeddings)} embeddings")

    print("\nFinding collision pairs...")
    pairs = find_collision_pairs(embeddings, limit=10)
    print(f"Found {len(pairs)} pairs to analyze")

    analyses = []
    for vid1, vid2, h in pairs:
        analysis = analyze_collision_pair(vid1, vid2)
        analyses.append(analysis)
        time.sleep(0.5)

    suggest_improvements(analyses)

    # Save for further analysis
    output_path = os.path.join(SCRIPT_DIR, 'collision_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"\nDetailed analysis saved to {output_path}")


if __name__ == '__main__':
    main()
