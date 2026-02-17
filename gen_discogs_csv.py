#!/usr/bin/env python3
"""Read the Discogs JSONL and generate videos_to_test_from_discogs.csv.

For each clique, emits all unique youtube_video URLs from the JSONL in
JSONL order, up to and including the last URL that appears in
videos_to_test.csv.  URLs after that cutoff are dropped.
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(SCRIPT_DIR, 'Discogs-VI-YT-20240701.jsonl')
VIDEOS_CSV = os.path.join(SCRIPT_DIR, 'videos_to_test.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'videos_to_test_from_discogs.csv')


def main():
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

                # Collect unique URLs in JSONL order
                seen = set()
                urls = []
                for v in data['versions']:
                    for t in v['tracks']:
                        for yt in t.get('youtube_video', []):
                            url = yt['url']
                            if url not in seen:
                                seen.add(url)
                                urls.append(url)

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


if __name__ == '__main__':
    main()
