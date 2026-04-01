"""
LIVI audio encoder: Whisper encoder + attention pooling + projection head.

Standalone module (no LIVI repo imports needed). Produces 768-dim
lyrics-informed embeddings for score-level fusion with CoverHunter.

Usage:
    from livi_model import init_livi_model, compute_livi_embedding

    livi, whisper = init_livi_model("/app/data/livi_checkpoint/livi.pth", device)
    embedding = compute_livi_embedding(livi, whisper, "song.wav", device)
"""

import contextlib
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import trunc_normal_

# --------------------------------------------------------------------------- #
# LIVI model architecture
# --------------------------------------------------------------------------- #


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class RotaryEmbedding(nn.Module):
    def __init__(self, embedding_dims, min_timescale=1, max_timescale=10000, dtype=torch.float32):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.compute_dtype = dtype
        half = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half)) / embedding_dims
        timescale = (min_timescale * (max_timescale / min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)

    def forward(self, inputs, position):
        position = position.unsqueeze(-1).unsqueeze(-1)
        sinusoid_inp = position / self.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)), dim=-1)


class Learned_Aggregation_Layer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.ql = nn.Linear(dim, dim, bias=qkv_bias)
        self.kl = nn.Linear(dim, dim, bias=qkv_bias)
        self.vl = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_prob = proj_drop
        self.rotary_emb = RotaryEmbedding(embedding_dims=head_dim)

    def forward(self, x):
        B, T, C = x.shape
        H = self.num_heads
        E = C // H
        q = self.ql(x[:, 0].unsqueeze(1)).view(B, 1, H, E)
        q = rearrange(q, "b t h e -> b h t e")
        k_pos = repeat(torch.arange(T), "l -> b l", b=B).to(x.device)
        k = self.kl(x).view(B, T, H, E)
        k = self.rotary_emb(k, position=k_pos)
        k = rearrange(k, "b t h e -> b h t e")
        v = self.vl(x).view(B, T, H, E)
        v = rearrange(v, "b t h e -> b h t e")
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, "b h t e -> b t (h e)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Layer_scale_init_Block_only_token(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, init_values=1e-4, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Learned_Aggregation_Layer(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))
        x_cls = x_cls + self.gamma_2 * self.mlp(self.norm2(x_cls))
        return x_cls


class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=2, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, init_scale=1e-4, rescale=0.02):
        super().__init__()
        self.attn = Layer_scale_init_Block_only_token(
            dim=int(dim), num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop, init_values=init_scale)
        self.norm_layer = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(dim)))
        self.rescale = rescale
        trunc_normal_(self.cls_token, std=self.rescale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        cls_token = self.attn(x, cls_token)
        x = torch.cat((cls_token, x), dim=1)
        x = self.norm_layer(x)
        return x[:, 0]


class Projection(nn.Module):
    def __init__(self, d_in=1024, d_out=768, hidden=None):
        super().__init__()
        hidden = hidden or [3072, 2048, 2048, 1536]
        layers, dim = [], d_in
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LayerNorm(h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LiviAudioEncoder(nn.Module):
    def __init__(self, dim_whisper=1280, dim_hiddens=None, dim_embed=768,
                 num_heads=1, mlp_ratio=2, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, init_scale=1e-4, **kwargs):
        super().__init__()
        dim_hiddens = dim_hiddens or [3072, 2048, 2048, 1536]
        self.pooling = AttentionPooling(
            dim=dim_whisper, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop, init_scale=init_scale)
        self.audio_proj = Projection(d_in=dim_whisper, d_out=dim_embed, hidden=dim_hiddens)

    def forward(self, audio):
        audio = self.pooling(audio)
        audio_embedding = self.audio_proj(audio)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)
        return audio_embedding


# --------------------------------------------------------------------------- #
# Whisper encoder wrapper
# --------------------------------------------------------------------------- #

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"


class WhisperEncoderWrapper:
    def __init__(self, model_name=WHISPER_MODEL_NAME, device=None):
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.device = device or torch.device("cpu")
        logging.info(f"Loading Whisper encoder ({model_name}) on {self.device}...")
        model = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.encoder = model.encoder.to(self.device).float()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        logging.info("Whisper encoder loaded.")

    def extract_mel(self, waveform_chunks):
        """Extract mel features from a list of numpy waveform chunks."""
        features = self.feature_extractor(
            waveform_chunks, sampling_rate=16000, padding=True,
            return_tensors="pt")
        return features.input_features.to(self.device)

    def encode(self, mel):
        """Run mel through frozen Whisper encoder -> hidden states."""
        use_autocast = self.device.type == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        with torch.no_grad(), autocast_ctx:
            hidden_states = self.encoder(mel).last_hidden_state
            return hidden_states.detach().float()


# --------------------------------------------------------------------------- #
# Audio loading & chunking
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16000
CHUNK_SEC = 30.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


def load_and_chunk_audio(wav_path):
    """Load audio at 16kHz mono, split into 30s non-overlapping chunks.
    Returns list of numpy arrays."""
    import librosa

    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

    T_total = waveform.shape[1]
    chunks = []
    for start in range(0, T_total, CHUNK_SAMPLES):
        end = min(start + CHUNK_SAMPLES, T_total)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < CHUNK_SAMPLES:
            chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[1]))
        chunks.append(chunk.squeeze(0).numpy())
    return chunks


# --------------------------------------------------------------------------- #
# Top-level API
# --------------------------------------------------------------------------- #

def init_livi_model(checkpoint_path, device):
    """Load LIVI audio encoder and Whisper encoder.

    Returns (livi_model, whisper_encoder).
    """
    livi = LiviAudioEncoder(
        dim_whisper=1280,
        dim_hiddens=[3072, 2048, 2048, 1536],
        dim_embed=768,
        num_heads=1,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        init_scale=1e-4,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict.pop("logit_scale", None)
    livi.load_state_dict(state_dict)
    livi.to(device)
    livi.eval()
    param_count = sum(p.numel() for p in livi.parameters()) / 1e6
    logging.info(f"Loaded LIVI audio encoder ({param_count:.1f}M params)")

    whisper = WhisperEncoderWrapper(device=device)
    return livi, whisper


def compute_livi_embedding(livi_model, whisper_encoder, wav_path, device):
    """Compute a 768-dim LIVI embedding from a WAV file.

    Returns numpy array of shape (768,), L2-normalized.
    """
    chunks = load_and_chunk_audio(wav_path)
    if not chunks:
        raise ValueError(f"No audio chunks from {wav_path}")

    mel = whisper_encoder.extract_mel(chunks)
    mel = mel.to(dtype=torch.float32, device=device)

    with torch.no_grad():
        hidden_states = whisper_encoder.encode(mel)
        embeddings = livi_model(hidden_states)  # (N_chunks, 768)
        embedding = embeddings.mean(dim=0)       # average over chunks

    return embedding.cpu().numpy().flatten()
