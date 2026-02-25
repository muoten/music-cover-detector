#!/usr/bin/env python3
"""
Pipeline for computing CoverHunter embeddings from iTunes 30-second previews.

This module provides functions for:
1. YouTube title lookup via oEmbed API
2. iTunes search and preview URL retrieval
3. Audio download and conversion
4. CQT feature extraction (matching CoverHunter training)
5. CoverHunter embedding computation

Usage:
    python pipeline.py [--checkpoint checkpoint/] [--output vectors_coverhunter.csv]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request

import numpy as np

# Add parent dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ============================================================================
# YouTube API helpers
# ============================================================================

def get_youtube_info(video_id):
    """Get YouTube video title and channel via oEmbed API."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {
                'title': data.get('title', ''),
                'channel': data.get('author_name', ''),
            }
    except Exception as e:
        return None


def get_youtube_metadata(video_id):
    """Get rich YouTube metadata via yt-dlp without downloading."""
    try:
        import yt_dlp
    except ImportError:
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'skip_download': True, 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', ''),
                'channel': info.get('channel', '') or info.get('uploader', ''),
                'artist': info.get('artist', ''),
                'track': info.get('track', ''),
            }
    except Exception:
        return None


def get_youtube_title(video_id):
    """Get YouTube video title via oEmbed API."""
    info = get_youtube_info(video_id)
    return info.get('title', '') if info else ''


def strip_diacritics(s):
    """Remove diacritics/accents from a string: Ângela → Angela, café → cafe."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def parse_artist_track(title):
    """Parse artist and track from YouTube title."""
    if not title:
        return None, None

    # Common separators
    for sep in [' - ', ' – ', ' — ', ' | ']:
        if sep in title:
            parts = title.split(sep, 1)
            artist = parts[0].strip()
            track = parts[1].strip()
            # Remove common suffixes
            for suffix in ['(Official Video)', '(Official Audio)', '(Lyrics)',
                          '[Official Video]', '[Official Audio]', '[Lyrics]',
                          '(Official Music Video)', '[Official Music Video]',
                          '(Audio)', '[Audio]', '(Video)', '[Video]',
                          '(Lyric Video)', '[Lyric Video]']:
                track = track.replace(suffix, '').strip()
            return artist, track

    return None, title


# ============================================================================
# iTunes API helpers
# ============================================================================

class ITunesRateLimitError(Exception):
    """Raised when iTunes API returns 429/403 after retries."""
    pass


# iTunes proxy URL: if set, routes iTunes API calls through a host-side proxy
# to avoid container-level rate limiting. Falls back to direct iTunes API.
ITUNES_PROXY_URL = os.environ.get('ITUNES_PROXY_URL', 'http://10.0.1.1:9090')


def search_itunes(term, limit=5):
    """Search iTunes for tracks via proxy (or direct fallback)."""
    encoded = urllib.parse.quote(term)
    # Try proxy first, fall back to direct iTunes API
    urls = []
    if ITUNES_PROXY_URL:
        urls.append(f"{ITUNES_PROXY_URL}/search?term={encoded}&media=music&entity=song&limit={limit}")
    urls.append(f"https://itunes.apple.com/search?term={encoded}&media=music&entity=song&limit={limit}")

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                return data.get('results', [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                if url == urls[-1]:
                    raise ITunesRateLimitError(f"iTunes rate limited ({e.code})")
                continue  # try next URL
            return []
        except Exception:
            if url != urls[-1]:
                continue  # try next URL
            return []
    return []


def search_itunes_detailed(term, limit=3):
    """Search iTunes and return full result details."""
    return search_itunes(term, limit)


def lookup_itunes_by_id(track_id):
    """Lookup iTunes track by ID."""
    url = f"https://itunes.apple.com/lookup?id={track_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])
            return results[0] if results else None
    except Exception as e:
        return None


def smart_search_itunes(youtube_id, title=None, channel=None, yt_metadata=None):
    """
    Smart iTunes search with fallback strategies.

    Returns: (track_info, track_id) or (None, None)
    """
    if not title:
        title = get_youtube_title(youtube_id)
    if not title:
        return None, None

    artist, track = parse_artist_track(title)

    def _names_match(a, b):
        """Check if two names match (substring in either direction, ignoring diacritics and spacing)."""
        a = strip_diacritics(a).lower()
        b = strip_diacritics(b).lower()
        if a in b or b in a:
            return True
        # Also compare with spaces collapsed (handles "McComb" vs "Mc Comb")
        a_compact = a.replace(' ', '')
        b_compact = b.replace(' ', '')
        return a_compact in b_compact or b_compact in a_compact

    # Extract artist from YouTube channel name
    channel_artist = None
    is_topic = False
    if channel:
        channel_artist = channel
        if channel_artist.endswith(' - Topic'):
            channel_artist = channel_artist[:-8]
            is_topic = True

    # Best known artist (from yt-dlp or channel) for final mismatch rejection
    known_artist = (yt_metadata.get('artist', '') if yt_metadata else '') or channel_artist

    def _artist_ok(itunes_artist):
        """Check if iTunes artist matches known artist."""
        if not known_artist:
            return True
        return _names_match(known_artist, itunes_artist)

    # Clean YouTube Music title: strip "- Original" suffix common on Topic channels
    clean_title = re.sub(r'\s*-\s*Original\s*$', '', title, flags=re.IGNORECASE).strip() or title

    def _clean_search_term(term):
        """Strip parenthetical/bracket suffixes that poison iTunes search.

        Removes: (Official), (Official Audio), (Official Video), (Remastered),
        (Remastered YYYY), (Live), [Official Audio], etc.
        """
        return re.sub(r'\s*[\(\[][^)\]]*(?:official|remaster|live|version|audio|video|mono|stereo)[^)\]]*[\)\]]',
                       '', term, flags=re.IGNORECASE).strip() or term

    # For channel artists with "&" or "and", extract primary artist for fallback searches
    primary_artist = channel_artist
    if channel_artist and re.search(r'\s+[&]\s+|\s+and\s+', channel_artist, re.IGNORECASE):
        primary_artist = re.split(r'\s+[&]\s+|\s+and\s+', channel_artist, flags=re.IGNORECASE)[0].strip()

    # Strategy YT: Use structured artist+track from yt-dlp metadata (highest confidence)
    if yt_metadata:
        yt_artist = yt_metadata.get('artist', '')
        yt_track = yt_metadata.get('track', '')
        if yt_artist and yt_track:
            for term in dict.fromkeys([f"{yt_artist} {yt_track}", _clean_search_term(f"{yt_artist} {yt_track}")]):
                results = search_itunes(term)
                for r in results:
                    if (_names_match(yt_artist, r.get('artistName', '')) and
                            _names_match(yt_track, r.get('trackName', '')) and
                            r.get('previewUrl')):
                        return r, r.get('trackId')

    # Strategy 0: Use channel artist + track (best for Topic channels)
    if channel_artist:
        # For Topic channels, the full title IS the track name (not parsed "track")
        expected_track = clean_title if is_topic else (track or clean_title)
        search_term = f"{channel_artist} {expected_track}"
        # Try: original, diacritics-stripped, cleaned (no parentheticals), primary artist only
        search_variants = [search_term, strip_diacritics(search_term), _clean_search_term(search_term)]
        if primary_artist != channel_artist:
            search_variants.append(f"{primary_artist} {expected_track}")
            search_variants.append(_clean_search_term(f"{primary_artist} {expected_track}"))
        for term in dict.fromkeys(search_variants):
            results = search_itunes(term)
            for r in results:
                if (_names_match(channel_artist, r.get('artistName', '')) and
                        _names_match(expected_track, r.get('trackName', '')) and
                        r.get('previewUrl')):
                    return r, r.get('trackId')

        # For Topic channels, Strategy 0 is authoritative — skip slower strategies
        if is_topic:
            return None, None

    # Strategy 1: Search with artist + track (from title parsing)
    if artist and track:
        for term in dict.fromkeys([f"{artist} {track}", _clean_search_term(f"{artist} {track}")]):
            results = search_itunes(term)
            if results:
                for r in results:
                    if (_names_match(artist, r.get('artistName', '')) and
                            _names_match(track, r.get('trackName', '')) and
                            r.get('previewUrl') and _artist_ok(r.get('artistName', ''))):
                        return r, r.get('trackId')
                for r in results:
                    if (r.get('previewUrl') and _artist_ok(r.get('artistName', '')) and
                            _names_match(track, r.get('trackName', ''))):
                        return r, r.get('trackId')

    # Strategy 2: Search with just title
    expected = track or clean_title
    for term in dict.fromkeys([title, _clean_search_term(title)]):
        results = search_itunes(term)
        for r in results:
            if (r.get('previewUrl') and _artist_ok(r.get('artistName', '')) and
                    _names_match(expected, r.get('trackName', ''))):
                return r, r.get('trackId')

    return None, None


# ============================================================================
# Audio processing helpers
# ============================================================================

def download_preview(url, output_path):
    """Download iTunes preview to file."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        return False


def convert_to_wav(input_path, output_path, sample_rate=16000):
    """Convert audio file to WAV with specified sample rate."""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', '1',
            '-f', 'wav',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        return False


def load_audio(wav_path, sample_rate=16000):
    """Load audio file using librosa."""
    import librosa
    signal, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    return signal


# ============================================================================
# CQT feature extraction (matching CoverHunter training)
# ============================================================================

def compute_cqt_features(signal, sample_rate=16000, hop_size=0.04, mean_size=3):
    """
    Compute CQT features matching CoverHunter training pipeline.

    Args:
        signal: Audio signal as numpy array
        sample_rate: Sample rate (16000 Hz for CoverHunter)
        hop_size: Hop size in seconds (0.04 = 25 frames/sec)
        mean_size: Downsampling factor (3 = ~8.3 frames/sec)

    Returns:
        cqt: CQT features as numpy array (time, freq=96)
    """
    from model.cqt import PyCqt, shorter

    # Create CQT extractor
    cqt_extractor = PyCqt(
        sample_rate=sample_rate,
        hop_size=hop_size,
        octave_resolution=12,
        min_freq=32,
        max_freq=sample_rate // 2,
    )

    # Compute CQT spectrogram
    cqt = cqt_extractor.compute_cqt(signal_float=signal, feat_dim_first=False)

    # Downsample with mean pooling
    if mean_size > 1:
        cqt = shorter(cqt, mean_size)

    return cqt.astype(np.float32)


# ============================================================================
# Embedding computation
# ============================================================================

def compute_embedding(model, cqt_features, device='cpu'):
    """
    Compute CoverHunter embedding from CQT features.

    Args:
        model: Loaded CoverHunter model
        cqt_features: CQT spectrogram (time, freq)
        device: Device to compute on

    Returns:
        embedding: 128-dim L2-normalized embedding
    """
    import torch

    # Prepare input tensor: (batch=1, time, freq)
    feat = torch.from_numpy(cqt_features).float().unsqueeze(0)
    feat = feat.to(device)

    # Run inference
    with torch.no_grad():
        embed, _ = model.inference(feat)

    # Convert to numpy and L2 normalize
    embed = embed.cpu().numpy().flatten()
    embed = embed / (np.linalg.norm(embed) + 1e-8)

    return embed


def process_youtube_video(youtube_id, model, device='cpu', temp_dir=None):
    """
    Process a single YouTube video: lookup -> iTunes -> preview -> CQT -> embedding.

    Args:
        youtube_id: YouTube video ID
        model: Loaded CoverHunter model
        device: Device to compute on
        temp_dir: Temporary directory for audio files

    Returns:
        dict with keys: success, embedding, track_id, error
    """
    import librosa

    result = {
        'success': False,
        'embedding': None,
        'track_id': None,
        'error': None,
    }

    # Get YouTube title
    title = get_youtube_title(youtube_id)
    if not title:
        result['error'] = 'Failed to get YouTube title'
        return result

    # Search iTunes
    track_info, track_id = smart_search_itunes(youtube_id, title)
    if not track_info or not track_info.get('previewUrl'):
        result['error'] = 'No iTunes preview found'
        return result

    result['track_id'] = track_id
    preview_url = track_info['previewUrl']

    # Download and process
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    m4a_path = os.path.join(temp_dir, f'{youtube_id}.m4a')
    wav_path = os.path.join(temp_dir, f'{youtube_id}.wav')

    try:
        # Download preview
        if not download_preview(preview_url, m4a_path):
            result['error'] = 'Failed to download preview'
            return result

        # Convert to WAV
        if not convert_to_wav(m4a_path, wav_path, sample_rate=16000):
            result['error'] = 'Failed to convert to WAV'
            return result

        # Load audio
        signal = load_audio(wav_path, sample_rate=16000)

        # Compute CQT features
        cqt = compute_cqt_features(signal, sample_rate=16000, hop_size=0.04, mean_size=3)

        # Compute embedding
        embedding = compute_embedding(model, cqt, device)

        result['success'] = True
        result['embedding'] = embedding

    except Exception as e:
        result['error'] = str(e)

    finally:
        # Clean up temp files
        for path in [m4a_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

    return result


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(input_file, output_file, checkpoint_dir, progress_file=None, device='cpu'):
    """
    Run the full pipeline on a list of YouTube IDs.

    Args:
        input_file: Path to file with YouTube IDs (one per line)
        output_file: Path to output CSV file
        checkpoint_dir: Path to CoverHunter checkpoint directory
        progress_file: Path to progress JSON file (for resumability)
        device: Device to use ('cpu', 'cuda', 'mps')
    """
    import torch
    from model.utils import load_model

    # Load model
    print(f"Loading CoverHunter model from {checkpoint_dir}...")
    model = load_model(checkpoint_dir, device)
    print(f"Model loaded on {device}")

    # Load progress if exists
    progress = {'processed': [], 'failed': {}, 'track_ids': {}}
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"Resuming from {len(progress['processed'])} processed videos")

    processed_set = set(progress['processed'])

    # Load YouTube IDs
    with open(input_file, 'r') as f:
        youtube_ids = [line.strip() for line in f if line.strip()]
    print(f"Total videos: {len(youtube_ids)}")

    # Open output file in append mode
    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    out_f = open(output_file, 'a')
    if write_header:
        out_f.write('youtube_id,embeddings\n')

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        for i, vid in enumerate(youtube_ids):
            if vid in processed_set:
                continue

            print(f"[{i+1}/{len(youtube_ids)}] Processing {vid}...", end=' ')

            result = process_youtube_video(vid, model, device, temp_dir)

            if result['success']:
                # Write to CSV
                emb_str = ' '.join(f'{x:.6f}' for x in result['embedding'])
                out_f.write(f'{vid},"[ {emb_str} ]"\n')
                out_f.flush()

                progress['processed'].append(vid)
                if result['track_id']:
                    progress['track_ids'][vid] = result['track_id']
                print(f"OK (trackId={result['track_id']})")
            else:
                progress['failed'][vid] = result['error']
                print(f"FAILED: {result['error']}")

            # Save progress periodically
            if progress_file and (len(progress['processed']) % 10 == 0):
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)

            # Rate limit
            time.sleep(0.2)

    finally:
        out_f.close()

        # Save final progress
        if progress_file:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

    # Print summary
    total = len(youtube_ids)
    success = len(progress['processed'])
    failed = len(progress['failed'])
    print(f"\nComplete: {success}/{total} successful, {failed} failed")


def main():
    # Default checkpoint: check persistent volume first, then local
    persistent_checkpoint = '/app/data/coverhunter_checkpoint'
    local_checkpoint = os.path.join(SCRIPT_DIR, 'checkpoint')
    default_checkpoint = persistent_checkpoint if os.path.exists(os.path.join(persistent_checkpoint, 'hparams.yaml')) else local_checkpoint

    parser = argparse.ArgumentParser(description='Compute CoverHunter embeddings from iTunes previews')
    parser.add_argument('--input', default='youtube_ids.txt', help='Input file with YouTube IDs')
    parser.add_argument('--output', default='vectors_coverhunter.csv', help='Output CSV file')
    parser.add_argument('--checkpoint', default=default_checkpoint, help='Checkpoint directory')
    parser.add_argument('--progress', default='progress.json', help='Progress file for resumability')
    parser.add_argument('--device', default='cpu', help='Device (cpu, cuda, mps)')
    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        checkpoint_dir=args.checkpoint,
        progress_file=args.progress,
        device=args.device,
    )


if __name__ == '__main__':
    main()
