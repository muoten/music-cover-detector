#!/usr/bin/env python3
"""
Extract YouTube IDs from vectors.csv file.

Usage:
    python extract_youtube_ids.py [--input ../vectors.csv] [--output youtube_ids.txt]
"""

import argparse
import os


def extract_ids(input_path, output_path):
    """Extract YouTube IDs from vectors.csv."""
    youtube_ids = []

    with open(input_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid_or_path = parts[0]
            # Extract YouTube ID from path like /tmp/.../ABC123.wav
            if '/' in vid_or_path:
                vid = os.path.basename(vid_or_path).replace('.wav', '')
            else:
                vid = vid_or_path
            # Validate it looks like a YouTube ID (11 chars, alphanumeric + - _)
            if len(vid) == 11:
                youtube_ids.append(vid)

    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for vid in youtube_ids:
        if vid not in seen:
            seen.add(vid)
            unique_ids.append(vid)

    with open(output_path, 'w') as f:
        for vid in unique_ids:
            f.write(vid + '\n')

    print(f"Extracted {len(unique_ids)} unique YouTube IDs from {input_path}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract YouTube IDs from vectors.csv')
    parser.add_argument('--input', default='../vectors.csv', help='Input vectors.csv file')
    parser.add_argument('--output', default='youtube_ids.txt', help='Output file with YouTube IDs')
    args = parser.parse_args()

    extract_ids(args.input, args.output)


if __name__ == '__main__':
    main()
