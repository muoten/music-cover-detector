#!/usr/bin/env python3
"""Read the Discogs JSONL and generate videos_to_test_from_discogs.csv.

Modes:
  (no flags)   Regenerate videos_to_test_from_discogs.csv from the JSONL,
               keeping only entries already in videos_to_test.csv (cutoff mode).

  --add N      Append N new URLs to videos_to_test.csv from cliques not yet
               present.  Only picks cliques with ≥2 new URLs (need pairs for
               cover detection).
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(SCRIPT_DIR, 'Discogs-VI-YT-20240701.jsonl')
VIDEOS_CSV = os.path.join(SCRIPT_DIR, 'videos_to_test.csv')
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


def regenerate():
    """Original cutoff mode: regenerate videos_to_test_from_discogs.csv."""
    # Step 1: Build a set of (clique, url) pairs from videos_to_test.csv
    print(f'Reading Discogs entries from {VIDEOS_CSV}...')
    discogs_pairs = set()
    with open(VIDEOS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['clique'].startswith('C-'):
                discogs_pairs.add((row['clique'], row['youtube_url']))
    print(f'  Found {len(discogs_pairs)} Discogs (clique, url) pairs')

    # Step 2: Walk the JSONL in order, for each clique collect unique URLs
    #         and emit all of them up to (and including) the last one that
    #         appears in videos_to_test.csv.
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

                # Find the index of the last URL present in videos_to_test
                last_match = -1
                for i, url in enumerate(urls):
                    if (cid, url) in discogs_pairs:
                        last_match = i

                if last_match == -1:
                    continue

                # Emit all URLs up to and including the last match
                for url in urls[:last_match + 1]:
                    writer.writerow([cid, url])
                    total += 1
                cliques_written += 1

    print(f'  Wrote {total} rows ({cliques_written} cliques)')


def add_new(target_count):
    """Append new URLs from unseen cliques to videos_to_test.csv."""
    # Step 1: Load all existing URLs from videos_to_test.csv
    print(f'Reading existing entries from {VIDEOS_CSV}...')
    existing_urls = set()
    existing_cliques = set()
    with open(VIDEOS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_urls.add(row['youtube_url'])
            if row['clique'].startswith('C-'):
                existing_cliques.add(row['clique'])
    print(f'  {len(existing_urls)} URLs from {len(existing_cliques)} Discogs cliques')

    # Step 2: Walk the JSONL and collect new URLs from new cliques
    print(f'Scanning {JSONL_PATH} for new cliques...')
    new_rows = []
    cliques_added = 0
    cliques_skipped_few_urls = 0

    with open(JSONL_PATH) as f:
        for line in f:
            if len(new_rows) >= target_count:
                break

            data = json.loads(line)
            cid = data['clique_id']

            # Skip cliques we already have
            if cid in existing_cliques:
                continue

            urls = extract_urls(data)

            # Filter out any URLs that happen to exist under a different clique
            new_urls = [u for u in urls if u not in existing_urls]

            # Need ≥2 new URLs to form a cover pair
            if len(new_urls) < 2:
                cliques_skipped_few_urls += 1
                continue

            # Take URLs up to the remaining budget
            remaining = target_count - len(new_rows)
            batch = new_urls[:remaining] if len(new_urls) > remaining else new_urls
            # Still need ≥2 after budget trimming
            if len(batch) < 2:
                break

            for url in batch:
                new_rows.append((cid, url))
            cliques_added += 1

    if not new_rows:
        print('  No new cliques with ≥2 URLs available.')
        return

    # Step 3: Append to videos_to_test.csv
    with open(VIDEOS_CSV, 'a', newline='') as out:
        writer = csv.writer(out)
        for cid, url in new_rows:
            writer.writerow([cid, url])

    print(f'  Added {len(new_rows)} URLs from {cliques_added} new cliques')
    print(f'  Skipped {cliques_skipped_few_urls} cliques with <2 new URLs')
    print(f'  New total: {len(existing_urls) + len(new_rows)} URLs')


def main():
    parser = argparse.ArgumentParser(description='Discogs CSV generator')
    parser.add_argument('--add', type=int, metavar='N',
                        help='Append N new URLs from unseen cliques to videos_to_test.csv')
    args = parser.parse_args()

    if args.add is not None:
        add_new(args.add)
    else:
        regenerate()


if __name__ == '__main__':
    main()
