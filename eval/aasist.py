# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import yaml
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ===== Path =====
PROTOCOL_FILE = "./PeerJ/protocol/ASVspoof2019.csv"
AUDIO_DIR1 = "./ASVspoof5/release_eval/flac_E_eval"
AUDIO_DIR2 = "./ASVspoof5/release_eval/flac_E_eval"   # sub path
OUTPUT_TXT = "./PeerJ/exp2/CM_aasist/aasist.txt"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.aasist import Model  # 클래스명: Model

MODEL_PATH = "./aasist/models/weights/AASIST.pth"
CONFIG_PATH = "./AASIST_conf.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== AASIST =====
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
model_config = config["model_config"]

model = Model(d_args=model_config, device=DEVICE).to(DEVICE)

# state_dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)

# DataParallel
model = torch.nn.DataParallel(model).to(DEVICE)
model.eval()

def load_and_prepare(wav_path, target_len=64600, target_sr=16000):
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Sample rate {sr} != {target_sr}")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if len(wav) >= target_len:
        wav = wav[:target_len]
    else:
        reps = target_len // len(wav) + 1
        wav = np.tile(wav, reps)[:target_len]
    return torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)

# ===== Read protocol =====
protocol_entries = []
with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        enroll_id, utt_id, label, trial_type = parts
        protocol_entries.append((enroll_id, utt_id, label, trial_type))

print(f"Num: {len(protocol_entries)}")

# ===== Scoring =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
skipped = []
results = []

with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")

    for enroll_id, utt_id, label, trial_type in tqdm(protocol_entries, desc="Scoring (AASIST)"):

        adv_path = os.path.join(AUDIO_DIR1, f"{utt_id}.flac")

        if not os.path.exists(adv_path):
            alt_path = os.path.join(AUDIO_DIR2, f"{utt_id}.wav")
            if os.path.exists(alt_path):
                adv_path = alt_path
            else:
                skipped.append((utt_id, f"File not found in both: {AUDIO_DIR1}, {AUDIO_DIR2}"))
                continue

        try:
            wav_tensor = load_and_prepare(adv_path).to(DEVICE)
            with torch.no_grad():
                out = model(wav_tensor)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                cm_score = logits[0, 1].item()

            fw.write(f"{utt_id},{enroll_id},{label},{trial_type},{cm_score:.6f}\n")
            results.append((utt_id, enroll_id, cm_score, label, trial_type))

        except Exception as e:
            skipped.append((adv_path, str(e)))
print(f"\n Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f" Skipped {len(skipped)} files. First few issues:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")
