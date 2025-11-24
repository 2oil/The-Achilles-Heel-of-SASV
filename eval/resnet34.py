# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== Path =====
PROTOCOL_FILE = "./PeerJ/protocol/ASVspoof5.csv"
AUDIO_DIR1 = "./ASVspoof5/release_eval/flac_E_eval"
AUDIO_DIR2 = "./PeerJ/exp2/CM_aasist/cm_bim_0007"   # sub dir
OUTPUT_TXT = "./PeerJ/exp2/CM_aasist/resnet34.txt"
ENROLL_ROOT = "./ASVspoof5/release_eval/flac_E_eval"

# ===== model import =====
sys.path.append("./")
from ResNetModels.ResNetSE34V2 import MainModel

MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load model =====
model = MainModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint.get("model", checkpoint)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("__S__."):
        new_key = k.replace("__S__.", "")
    elif k.startswith("__L__."):
        continue
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=True)
model.eval()

# ===== utils =====
def read_mono_float32(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav

@torch.no_grad()
def extract_embedding(wav_path):
    wav = read_mono_float32(wav_path)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    emb = model(wav_tensor).squeeze().detach().cpu().numpy()
    return emb

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

protocol_entries = []
with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        enroll_id, utt_id, label, trial_type = parts
        protocol_entries.append((enroll_id, utt_id, label, trial_type))

print(f"Num: {len(protocol_entries)}")

embeddings = {}
enr_embeddings = {}

# ===== Scoring =====
results = []
skipped = []

for enroll_id, utt_id, label, trial_type in tqdm(protocol_entries, desc="Scoring from protocol"):
    adv_path = os.path.join(AUDIO_DIR1, f"{utt_id}.flac")
    if not os.path.exists(adv_path):
        alt_path = os.path.join(AUDIO_DIR2, f"{utt_id}.wav")
        if os.path.exists(alt_path):
            adv_path = alt_path
        else:
            skipped.append((utt_id, f"Audio not found in both dirs: {AUDIO_DIR1}, {AUDIO_DIR2}"))
            continue

    enroll_path = find_enroll_path(enroll_id)
    if not enroll_path or not os.path.exists(enroll_path):
        skipped.append((utt_id, f"Enroll audio not found for {enroll_id}"))
        continue

    try:
        if adv_path not in embeddings:
            embeddings[adv_path] = extract_embedding(adv_path)
        if enroll_path not in enr_embeddings:
            enr_embeddings[enroll_path] = extract_embedding(enroll_path)

        s = cosine(embeddings[adv_path], enr_embeddings[enroll_path])
        results.append((utt_id, enroll_id, s, label, trial_type))
    except Exception as e:
        skipped.append((utt_id, str(e)))

# ===== Save results =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,cosine_score,label,trial_type\n")
    for utt_id, enroll_id, s, label, trial_type in results:
        fw.write(f"{utt_id},{enroll_id},{s:.6f},{label},{trial_type}\n")

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} entries. First few issues:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")

