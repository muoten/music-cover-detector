#!/usr/bin/env python3
"""
Song crawler for indexing unprocessed videos from videos_to_test.csv.

Calls the API's /api/search endpoint for each unindexed song.
Pre-filters already-indexed songs (from vectors.csv) and previously
attempted songs (from crawl_progress.txt) to avoid wasted API calls.

Usage:
    python crawl_songs.py --api http://localhost:8080 --delay 2
"""

import argparse
import csv
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)


def resolve_path(name, persistent_dir='/app/data', local_dir='.'):
    """Resolve a file path: check persistent volume first, fall back to local."""
    persistent = os.path.join(persistent_dir, name)
    if os.path.exists(persistent) or os.path.isdir(persistent_dir):
        return persistent
    return os.path.join(local_dir, name)


def load_indexed_ids(vectors_path):
    """Load video IDs already in the embeddings database."""
    indexed = set()
    if not os.path.exists(vectors_path):
        logging.warning(f"vectors.csv not found at {vectors_path}")
        return indexed

    with open(vectors_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if not parts:
                continue
            vid_or_path = parts[0]
            if '/' in vid_or_path:
                vid = os.path.basename(vid_or_path).replace('.wav', '')
            else:
                vid = vid_or_path
            if vid:
                indexed.add(vid)

    return indexed


def load_progress(progress_path):
    """Load previously attempted video IDs from progress file."""
    attempted = set()
    if not os.path.exists(progress_path):
        return attempted

    with open(progress_path, 'r') as f:
        for line in f:
            vid = line.strip()
            if vid:
                attempted.add(vid)

    return attempted


def mark_attempted(progress_path, video_id):
    """Append a video ID to the progress file."""
    with open(progress_path, 'a') as f:
        f.write(video_id + '\n')


def load_videos_to_test(csv_path):
    """Load video IDs from videos_to_test.csv."""
    videos = []
    if not os.path.exists(csv_path):
        logging.error(f"videos_to_test.csv not found at {csv_path}")
        return videos

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            url = row[1].strip()
            match = re.search(r'v=([a-zA-Z0-9_-]{11})', url)
            if match:
                videos.append(match.group(1))

    return videos


def load_track_ids():
    """Load track_id mappings and find video IDs with duplicate track_ids."""
    all_track_ids = {}

    for name in ['track_ids.json', 'track_ids_new.json']:
        path = resolve_path(name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_track_ids.update(json.load(f))

    # Find track_ids used by more than one video
    tid_to_vids = {}
    for vid, tid in all_track_ids.items():
        tid_to_vids.setdefault(tid, []).append(vid)

    duplicates = set()
    for tid, vids in tid_to_vids.items():
        if len(vids) > 1:
            duplicates.update(vids)

    return all_track_ids, duplicates


def report_progress(api_url, phase, processed, total, rate_per_hour=0, eta_hours=0, remaining=0, dup_total=0, new_total=0, no_tid_total=0):
    """POST crawl progress to the API (silent on failure)."""
    try:
        payload = json.dumps({
            'phase': phase, 'processed': processed, 'total': total,
            'remaining': remaining, 'rate_per_hour': round(rate_per_hour),
            'eta_hours': round(eta_hours, 1),
            'dup_total': dup_total, 'new_total': new_total,
            'no_tid_total': no_tid_total,
        }).encode('utf-8')
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/crawl-progress",
            data=payload, headers={'Content-Type': 'application/json'},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logging.warning(f"Failed to report progress: {e}")


def call_search_api(api_url, video_id, force=False):
    """
    Call the /api/search endpoint for a video ID.

    Returns: (success, is_transient_error, message)
      - success: True if song was processed (added or already in DB)
      - is_transient_error: True if error is transient (should retry)
      - message: description of result
    """
    url = f"{api_url.rstrip('/')}/api/search"
    payload = json.dumps({
        'url': f'https://www.youtube.com/watch?v={video_id}',
        'top_k': 1,
        'force': force,
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            in_db = data.get('query', {}).get('in_database', False)
            if in_db:
                return True, False, 'already in database'
            else:
                return True, False, 'added to database'
    except urllib.error.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', errors='replace')
        except Exception:
            pass

        if e.code == 404:
            # Permanent failure (no iTunes preview, etc.)
            error_detail = ''
            try:
                error_data = json.loads(body)
                error_detail = error_data.get('error', '')
            except Exception:
                error_detail = body[:200]
            return False, False, f'HTTP 404: {error_detail}'
        elif e.code >= 500:
            return False, True, f'HTTP {e.code}: {body[:200]}'
        else:
            # Other 4xx errors are permanent
            return False, False, f'HTTP {e.code}: {body[:200]}'
    except urllib.error.URLError as e:
        return False, True, f'URL error: {e.reason}'
    except TimeoutError:
        return False, True, 'Request timed out'
    except Exception as e:
        return False, True, f'Unexpected error: {e}'


def call_cleanup_api(api_url):
    """POST to /api/cleanup-unverified to remove tracks with no iTunes match."""
    try:
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/cleanup-unverified",
            data=b'{}', headers={'Content-Type': 'application/json'},
            method='POST',
        )
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode('utf-8'))
        logging.info(f"Cleanup API: removed {result.get('removed', '?')} unverified, "
                     f"{result.get('total_remaining', '?')} remaining")
        return result
    except Exception as e:
        logging.error(f"Cleanup API call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Crawl unindexed songs')
    parser.add_argument('--api', default='https://coverdetector.com',
                        help='API base URL (default: https://coverdetector.com)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between requests in seconds (default: 2)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Max retries for transient errors (default: 3)')
    args = parser.parse_args()

    # Resolve file paths
    vectors_path = resolve_path('vectors.csv')
    progress_path = resolve_path('crawl_progress.txt')

    csv_paths = ['/app/videos_to_test.csv', './videos_to_test.csv']
    csv_path = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if not csv_path:
        logging.error("Cannot find videos_to_test.csv")
        return

    # Load data
    logging.info(f"Loading indexed IDs from {vectors_path}...")
    indexed = load_indexed_ids(vectors_path)
    logging.info(f"Indexed songs: {len(indexed)}")

    logging.info(f"Loading progress from {progress_path}...")
    attempted = load_progress(progress_path)
    logging.info(f"Previously attempted: {len(attempted)}")

    resolved_path = resolve_path('duplicates_resolved.txt')
    resolved = load_progress(resolved_path)
    logging.info(f"Previously resolved duplicates: {len(resolved)}")

    no_trackid_resolved_path = resolve_path('no_trackid_resolved.txt')
    no_trackid_resolved = load_progress(no_trackid_resolved_path)
    logging.info(f"Previously resolved no-trackid: {len(no_trackid_resolved)}")

    logging.info(f"Loading videos from {csv_path}...")
    all_videos = load_videos_to_test(csv_path)
    logging.info(f"Total videos in CSV: {len(all_videos)}")

    # Find songs with duplicate track_ids to re-crawl
    logging.info("Loading track_ids to find duplicates...")
    all_track_ids, duplicate_vids = load_track_ids()
    logging.info(f"Songs with duplicate track_ids: {len(duplicate_vids)}")

    # Load removed IDs to exclude from all queues
    removed_path = resolve_path('removed.txt')
    removed = load_progress(removed_path)
    if removed:
        logging.info(f"Previously removed (excluded): {len(removed)}")

    # Build work queue: skip indexed, already-attempted, and removed
    skip = indexed | attempted | removed
    queue = [vid for vid in all_videos if vid not in skip]

    # Build no-trackid re-crawl queue (indexed songs with no track_id mapping)
    track_id_keys = set(all_track_ids.keys())
    no_trackid_queue = [vid for vid in all_videos if vid in indexed and vid not in track_id_keys and vid not in no_trackid_resolved and vid not in removed]
    logging.info(f"Songs with no track_id: {len(no_trackid_queue)}")

    # Build duplicate re-crawl queue (indexed songs with duplicate track_ids, skip already resolved)
    duplicate_queue = [vid for vid in all_videos if vid in duplicate_vids and vid in indexed and vid not in resolved and vid not in removed]
    logging.info(f"Songs to process: {len(no_trackid_queue)} unverified + {len(duplicate_queue)} duplicates + {len(queue)} new")

    if not queue and not duplicate_queue and not no_trackid_queue:
        logging.info("Nothing to crawl.")
        return

    start_time = time.time()
    last_report_time = start_time
    total_work = len(no_trackid_queue) + len(duplicate_queue) + len(queue)
    report_progress(args.api, 'crawling', 0, total_work, remaining=total_work,
                    dup_total=len(duplicate_queue), new_total=len(queue), no_tid_total=len(no_trackid_queue))

    # Phase 0: Re-crawl songs with no track_id (unverified iTunes matches)
    if no_trackid_queue:
        logging.info(f"=== Phase 0: Re-crawling {len(no_trackid_queue)} songs with no track_id ===")
        ntid_stats = {'updated': 0, 'errors': 0}

        for i, video_id in enumerate(no_trackid_queue):
            retries = 0
            while retries <= args.max_retries:
                success, is_transient, message = call_search_api(args.api, video_id, force=True)

                if success:
                    ntid_stats['updated'] += 1
                    mark_attempted(no_trackid_resolved_path, video_id)
                    logging.info(f"[no_tid {i+1}/{len(no_trackid_queue)}] {video_id}: {message}")
                    break
                elif is_transient:
                    retries += 1
                    if retries <= args.max_retries:
                        wait = args.delay * (2 ** retries)
                        logging.warning(
                            f"[no_tid {i+1}/{len(no_trackid_queue)}] {video_id}: transient error "
                            f"(retry {retries}/{args.max_retries} in {wait:.0f}s): {message}"
                        )
                        time.sleep(wait)
                    else:
                        ntid_stats['errors'] += 1
                        logging.error(
                            f"[no_tid {i+1}/{len(no_trackid_queue)}] {video_id}: giving up: {message}"
                        )
                        break
                else:
                    ntid_stats['errors'] += 1
                    mark_attempted(no_trackid_resolved_path, video_id)
                    logging.info(f"[no_tid {i+1}/{len(no_trackid_queue)}] {video_id}: failed — {message}")
                    break

            now = time.time()
            if now - last_report_time >= 20:
                ntid_processed = ntid_stats['updated'] + ntid_stats['errors']
                elapsed = now - start_time
                rate = ntid_processed / elapsed * 3600 if elapsed > 0 else 0
                remaining = len(no_trackid_queue) - (i + 1) + len(duplicate_queue) + len(queue)
                eta_hours = remaining / rate if rate > 0 else 0
                report_progress(args.api, 'no_trackid', ntid_processed, total_work, rate, eta_hours, remaining,
                                dup_total=len(duplicate_queue), new_total=len(queue), no_tid_total=len(no_trackid_queue))
                last_report_time = now
            time.sleep(args.delay)

        logging.info(
            f"=== No-trackid re-crawl done: {ntid_stats['updated']} updated | "
            f"{ntid_stats['errors']} errors ==="
        )

        # Remove tracks that still have no iTunes match after re-crawl
        logging.info("Cleaning up unverified tracks...")
        call_cleanup_api(args.api)

    # Phase 1: Re-crawl duplicates with force=True (first, so we can trace results quickly)
    if duplicate_queue:
        logging.info(f"=== Phase 1: Re-crawling {len(duplicate_queue)} songs with duplicate track_ids ===")
        dup_stats = {'updated': 0, 'unchanged': 0, 'errors': 0}

        for i, video_id in enumerate(duplicate_queue):
            retries = 0
            while retries <= args.max_retries:
                success, is_transient, message = call_search_api(args.api, video_id, force=True)

                if success:
                    dup_stats['updated'] += 1
                    mark_attempted(resolved_path, video_id)
                    logging.info(f"[dup {i+1}/{len(duplicate_queue)}] {video_id}: {message}")
                    break
                elif is_transient:
                    retries += 1
                    if retries <= args.max_retries:
                        wait = args.delay * (2 ** retries)
                        logging.warning(
                            f"[dup {i+1}/{len(duplicate_queue)}] {video_id}: transient error "
                            f"(retry {retries}/{args.max_retries} in {wait:.0f}s): {message}"
                        )
                        time.sleep(wait)
                    else:
                        dup_stats['errors'] += 1
                        logging.error(
                            f"[dup {i+1}/{len(duplicate_queue)}] {video_id}: giving up: {message}"
                        )
                        break
                else:
                    dup_stats['errors'] += 1
                    logging.info(f"[dup {i+1}/{len(duplicate_queue)}] {video_id}: failed — {message}")
                    break

            now = time.time()
            if now - last_report_time >= 20:
                dup_processed = dup_stats['updated'] + dup_stats['errors']
                elapsed = now - start_time
                rate = dup_processed / elapsed * 3600 if elapsed > 0 else 0
                remaining = len(duplicate_queue) - (i + 1) + len(queue)
                eta_hours = remaining / rate if rate > 0 else 0
                report_progress(args.api, 'dedup', len(no_trackid_queue) + dup_processed, total_work, rate, eta_hours, remaining,
                                dup_total=len(duplicate_queue), new_total=len(queue), no_tid_total=len(no_trackid_queue))
                last_report_time = now
            time.sleep(args.delay)

        logging.info(
            f"=== Duplicate re-crawl done: {dup_stats['updated']} updated | "
            f"{dup_stats['errors']} errors ==="
        )

    # Phase 2: Crawl new songs
    stats = {'added': 0, 'skipped': 0, 'errors': 0, 'transient_errors': 0}

    for i, video_id in enumerate(queue):
        # Re-check in case the API added it during a previous call's async processing
        if video_id in attempted:
            continue

        retries = 0
        while retries <= args.max_retries:
            success, is_transient, message = call_search_api(args.api, video_id)

            if success:
                mark_attempted(progress_path, video_id)
                attempted.add(video_id)
                stats['added'] += 1
                logging.info(f"[{i+1}/{len(queue)}] {video_id}: {message}")
                break
            elif is_transient:
                retries += 1
                if retries <= args.max_retries:
                    wait = args.delay * (2 ** retries)
                    logging.warning(
                        f"[{i+1}/{len(queue)}] {video_id}: transient error "
                        f"(retry {retries}/{args.max_retries} in {wait:.0f}s): {message}"
                    )
                    time.sleep(wait)
                else:
                    # Exhausted retries — don't mark as attempted so we retry next run
                    stats['transient_errors'] += 1
                    logging.error(
                        f"[{i+1}/{len(queue)}] {video_id}: giving up after "
                        f"{args.max_retries} retries: {message}"
                    )
                    break
            else:
                # Permanent failure — mark as attempted
                mark_attempted(progress_path, video_id)
                attempted.add(video_id)
                stats['skipped'] += 1
                logging.info(f"[{i+1}/{len(queue)}] {video_id}: skipped — {message}")
                break

        # Periodic stats
        processed = stats['added'] + stats['skipped'] + stats['transient_errors']
        now = time.time()
        elapsed = now - start_time
        rate = processed / elapsed * 3600 if elapsed > 0 else 0
        remaining = len(queue) - (i + 1)
        eta_hours = remaining / rate if rate > 0 else 0
        if processed > 0 and processed % 100 == 0:
            logging.info(
                f"--- Stats: {processed} processed | "
                f"{stats['added']} added | {stats['skipped']} skipped | "
                f"{stats['transient_errors']} transient errors | "
                f"{remaining} remaining | "
                f"{rate:.0f}/hr | ETA: {eta_hours:.1f}h"
            )
        if now - last_report_time >= 20:
            report_progress(args.api, 'crawling', len(no_trackid_queue) + len(duplicate_queue) + processed,
                            total_work, rate, eta_hours, remaining,
                            dup_total=len(duplicate_queue), new_total=len(queue), no_tid_total=len(no_trackid_queue))
            last_report_time = now

        time.sleep(args.delay)

    # Final stats
    elapsed = time.time() - start_time
    total = stats['added'] + stats['skipped'] + stats['transient_errors']
    logging.info(
        f"=== Crawl complete in {elapsed/3600:.1f}h === "
        f"{total} new processed | {stats['added']} added | "
        f"{stats['skipped']} skipped | {stats['transient_errors']} transient errors"
    )
    report_progress(args.api, 'done', total_work, total_work,
                    dup_total=len(duplicate_queue), new_total=len(queue), no_tid_total=len(no_trackid_queue))


if __name__ == '__main__':
    main()
