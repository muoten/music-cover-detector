#!/usr/bin/env python3
"""Generate and grow videos_to_test.csv from the Discogs JSONL.

Modes:
  --generate   Regenerate videos_to_test.csv from discogs_cliques.txt + JSONL.
               Deterministic: same inputs always produce the same CSV.

  --add N      Append N new URLs (from new cliques with ≥2 URLs) to both
               discogs_cliques.txt and videos_to_test.csv.

  (no flags)   Legacy: regenerate videos_to_test_from_discogs.csv using the
               cutoff method (for diffing/validation only).

Source of truth:
  discogs_cliques.txt  — "CLIQUE_ID URL_COUNT" per line (committed to git)
  Discogs-VI-YT-20240701.jsonl — full dataset (download from Zenodo record 13983028)
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(SCRIPT_DIR, 'Discogs-VI-YT-20240701.jsonl')
VIDEOS_CSV = os.path.join(SCRIPT_DIR, 'videos_to_test.csv')
CLIQUES_TXT = os.path.join(SCRIPT_DIR, 'discogs_cliques.txt')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'videos_to_test_from_discogs.csv')


def extract_urls(data):
    """Extract unique YouTube URLs from a JSONL clique entry, in order."""
    seen = set()
    urls = []
    for v in data['versions']:
        for t in v['tracks']:
            for yt in t.get('youtube_video', []):
                url = yt['url']
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls


def load_cliques():
    """Load the ordered list of (clique_id, url_count) tuples."""
    entries = []
    with open(CLIQUES_TXT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cid = parts[0]
            count = int(parts[1]) if len(parts) > 1 else None
            entries.append((cid, count))
    return entries


def generate():
    """Regenerate videos_to_test.csv from discogs_cliques.txt + JSONL."""
    entries = load_cliques()
    clique_counts = {cid: count for cid, count in entries}
    print(f'Loaded {len(entries)} cliques from {CLIQUES_TXT}')

    # Walk JSONL and collect URLs for included cliques
    print(f'Reading {JSONL_PATH}...')
    clique_urls = {}  # cid -> [urls]
    with open(JSONL_PATH) as f:
        for line in f:
            data = json.loads(line)
            cid = data['clique_id']
            if cid in clique_counts:
                clique_urls[cid] = extract_urls(data)

    missing = set(clique_counts) - set(clique_urls)
    if missing:
        print(f'  WARNING: {len(missing)} cliques not found in JSONL')

    # Write CSV in cliques.txt order, respecting url_count
    print(f'Writing {VIDEOS_CSV}...')
    total = 0
    with open(VIDEOS_CSV, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['clique', 'youtube_url'])
        for cid, count in entries:
            urls = clique_urls.get(cid, [])
            if count is not None:
                urls = urls[:count]
            for url in urls:
                writer.writerow([cid, url])
                total += 1

    print(f'  Wrote {total} rows ({len(entries)} cliques)')


def add_new(target_count):
    """Append new cliques from the JSONL to discogs_cliques.txt and videos_to_test.csv."""
    entries = load_cliques()
    existing_cliques = {cid for cid, _ in entries}
    print(f'Loaded {len(entries)} existing cliques from {CLIQUES_TXT}')

    existing_urls = set()
    with open(VIDEOS_CSV) as f:
        for row in csv.DictReader(f):
            existing_urls.add(row['youtube_url'])
    print(f'  {len(existing_urls)} existing URLs in {VIDEOS_CSV}')

    # Walk the JSONL and collect new cliques
    print(f'Scanning {JSONL_PATH} for new cliques...')
    new_rows = []       # (cid, url) pairs to append to CSV
    new_cliques = []    # (cid, url_count) to append to cliques.txt
    cliques_skipped = 0

    with open(JSONL_PATH) as f:
        for line in f:
            if len(new_rows) >= target_count:
                break

            data = json.loads(line)
            cid = data['clique_id']

            if cid in existing_cliques:
                continue

            urls = extract_urls(data)
            new_urls = [u for u in urls if u not in existing_urls]

            if len(new_urls) < 2:
                cliques_skipped += 1
                continue

            # Budget check
            remaining = target_count - len(new_rows)
            batch = new_urls[:remaining] if len(new_urls) > remaining else new_urls
            if len(batch) < 2:
                break

            for url in batch:
                new_rows.append((cid, url))
            new_cliques.append((cid, len(batch)))

    if not new_rows:
        print('  No new cliques with ≥2 URLs available.')
        return

    # Append to both files
    with open(CLIQUES_TXT, 'a') as f:
        for cid, count in new_cliques:
            f.write(f'{cid} {count}\n')

    with open(VIDEOS_CSV, 'a', newline='') as out:
        writer = csv.writer(out)
        for cid, url in new_rows:
            writer.writerow([cid, url])

    print(f'  Added {len(new_rows)} URLs from {len(new_cliques)} new cliques')
    if cliques_skipped:
        print(f'  Skipped {cliques_skipped} cliques with <2 new URLs')
    print(f'  New total: {len(existing_urls) + len(new_rows)} URLs ({len(entries) + len(new_cliques)} cliques)')


def regenerate_legacy():
    """Legacy cutoff mode: regenerate videos_to_test_from_discogs.csv."""
    print(f'Reading Discogs entries from {VIDEOS_CSV}...')
    discogs_pairs = set()
    with open(VIDEOS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['clique'].startswith('C-'):
                discogs_pairs.add((row['clique'], row['youtube_url']))
    print(f'  Found {len(discogs_pairs)} Discogs (clique, url) pairs')

    print(f'Reading {JSONL_PATH} and writing {OUTPUT_CSV}...')
    total = 0
    cliques_written = 0
    with open(OUTPUT_CSV, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['clique', 'youtube_url'])

        with open(JSONL_PATH) as f:
            for line in f:
                data = json.loads(line)
                cid = data['clique_id']
                urls = extract_urls(data)

                last_match = -1
                for i, url in enumerate(urls):
                    if (cid, url) in discogs_pairs:
                        last_match = i

                if last_match == -1:
                    continue

                for url in urls[:last_match + 1]:
                    writer.writerow([cid, url])
                    total += 1
                cliques_written += 1

    print(f'  Wrote {total} rows ({cliques_written} cliques)')


def main():
    parser = argparse.ArgumentParser(
        description='Generate and grow videos_to_test.csv from Discogs JSONL')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--generate', action='store_true',
                       help='Regenerate videos_to_test.csv from discogs_cliques.txt + JSONL')
    group.add_argument('--add', type=int, metavar='N',
                       help='Append N new URLs from unseen cliques')
    args = parser.parse_args()

    if args.generate:
        generate()
    elif args.add is not None:
        add_new(args.add)
    else:
        regenerate_legacy()


if __name__ == '__main__':
    main()
