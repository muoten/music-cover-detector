#!/usr/bin/env python3
"""
Smart iTunes matching that uses YouTube channel name as artist fallback.
Tests on collision pairs with different cliques (wrong matches).
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import hashlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline import parse_artist_track, STRIP_PATTERNS


def get_youtube_info(youtube_id):
    """Get video title AND channel name via oEmbed."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return {
                'title': data.get('title', ''),
                'channel': data.get('author_name', ''),
            }
    except:
        return None


def search_itunes(query, limit=5):
    """Search iTunes."""
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
    except:
        return []


def smart_search_strategies(title, channel):
    """Generate multiple search strategies."""
    artist, track = parse_artist_track(title)

    strategies = []

    # Strategy 1: Original (artist - track from title)
    if artist and track:
        strategies.append(('title_parsed', f'{artist} {track}'))

    # Strategy 2: Use channel as artist + track from title
    if track and channel:
        # Clean channel name (remove "- Topic", "VEVO", etc.)
        clean_channel = re.sub(r'\s*-\s*Topic$', '', channel, flags=re.IGNORECASE)
        clean_channel = re.sub(r'\s*VEVO$', '', clean_channel, flags=re.IGNORECASE)
        strategies.append(('channel_artist', f'{clean_channel} {track}'))

    # Strategy 3: Full title with channel
    if channel:
        clean_title = title
        for pattern in STRIP_PATTERNS:
            clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE)
        strategies.append(('channel_title', f'{channel} {clean_title.strip()}'))

    # Strategy 4: Just the track (fallback)
    if track:
        strategies.append(('track_only', track))

    return strategies


def score_match(itunes_result, youtube_title, youtube_channel):
    """Score how well an iTunes result matches the YouTube video."""
    itunes_artist = itunes_result.get('artistName', '').lower()
    itunes_track = itunes_result.get('trackName', '').lower()
    yt_title = youtube_title.lower()
    yt_channel = youtube_channel.lower()

    score = 0

    # Check if iTunes artist matches YouTube channel
    if itunes_artist in yt_channel or yt_channel in itunes_artist:
        score += 50

    # Check if iTunes track is in YouTube title
    if itunes_track in yt_title:
        score += 30

    # Check if iTunes artist is in YouTube title
    if itunes_artist in yt_title:
        score += 20

    return score


def find_best_match(youtube_id):
    """Find the best iTunes match for a YouTube video using smart strategies."""
    info = get_youtube_info(youtube_id)
    if not info:
        return None, None, 'oembed_fail'

    title = info['title']
    channel = info['channel']

    strategies = smart_search_strategies(title, channel)

    best_result = None
    best_score = -1
    best_strategy = None

    for strategy_name, query in strategies:
        time.sleep(0.2)
        results = search_itunes(query)

        for result in results[:3]:
            score = score_match(result, title, channel)
            if score > best_score:
                best_score = score
                best_result = result
                best_strategy = strategy_name

    if best_result:
        return best_result, best_strategy, None
    return None, None, 'no_match'


def compute_hash(emb):
    return hashlib.md5(emb.astype(np.float32).tobytes()).hexdigest()[:8]


def load_wrong_collision_pairs(limit=20):
    """Load collision pairs that are from different cliques (wrong matches)."""
    # Load embeddings
    embeddings = {}
    with open(os.path.join(SCRIPT_DIR, 'vectors_vinet.csv')) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                vid = parts[0]
                vec_str = parts[1].strip('"').strip('[]')
                try:
                    vec = np.array([float(x) for x in vec_str.split()])
                    if len(vec) == 512:
                        embeddings[vid] = vec
                except:
                    pass

    # Load clique map
    clique_map = {}
    with open(os.path.join(SCRIPT_DIR, 'videos_to_test.csv')) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and 'v=' in parts[1]:
                vid = parts[1].split('v=')[1].split('&')[0]
                clique_map[vid] = parts[0]

    # Find collision groups
    hash_to_vids = {}
    for vid, emb in embeddings.items():
        h = compute_hash(emb)
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    # Find wrong collision pairs (different cliques)
    wrong_pairs = []
    for h, vids in hash_to_vids.items():
        if len(vids) < 2:
            continue
        for i in range(len(vids)):
            for j in range(i+1, len(vids)):
                c1 = clique_map.get(vids[i])
                c2 = clique_map.get(vids[j])
                if c1 and c2 and c1 != c2:
                    wrong_pairs.append((vids[i], vids[j], c1, c2))
                    if len(wrong_pairs) >= limit:
                        return wrong_pairs

    return wrong_pairs


def main():
    print("Loading wrong collision pairs (different cliques)...")
    pairs = load_wrong_collision_pairs(limit=10)
    print(f"Found {len(pairs)} pairs to analyze\n")

    improved = 0
    still_same = 0

    for vid1, vid2, clique1, clique2 in pairs:
        print(f"{'='*60}")
        print(f"Pair: {vid1} (clique {clique1}) vs {vid2} (clique {clique2})")

        # Get info for both
        info1 = get_youtube_info(vid1)
        info2 = get_youtube_info(vid2)

        if not info1 or not info2:
            print("  Could not get YouTube info")
            continue

        print(f"  Video 1: '{info1['title']}' by {info1['channel']}")
        print(f"  Video 2: '{info2['title']}' by {info2['channel']}")

        # Find best matches
        time.sleep(0.3)
        match1, strategy1, err1 = find_best_match(vid1)
        time.sleep(0.3)
        match2, strategy2, err2 = find_best_match(vid2)

        if match1:
            print(f"  Match 1 ({strategy1}): {match1['artistName']} - {match1['trackName']}")
            print(f"           trackId: {match1.get('trackId')}")
        if match2:
            print(f"  Match 2 ({strategy2}): {match2['artistName']} - {match2['trackName']}")
            print(f"           trackId: {match2.get('trackId')}")

        if match1 and match2:
            if match1.get('trackId') != match2.get('trackId'):
                print(f"  ✓ DIFFERENT iTunes tracks! Smart matching could help.")
                improved += 1
            else:
                print(f"  ⚠ Still same iTunes track")
                still_same += 1

        print()

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Pairs that would improve with smart matching: {improved}")
    print(f"  Pairs still matching same track: {still_same}")
    print(f"  Improvement potential: {improved/(improved+still_same)*100:.1f}%" if improved+still_same > 0 else "N/A")


if __name__ == '__main__':
    main()
