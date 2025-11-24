import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== Path =====
AUDIO_DIR   = "./LA/ASVspoof2019_LA_eval/flac"
AUDIO_DIR2  = "./PeerJ/exp1/CM_aasist3/cm_bim_0007"
OUTPUT_TXT  = "./PeerJ/exp1/CM_aasist3/ecapa_tdnn.txt"
PROTOCOL_FILE = "./PeerJ/protocol/ASVspoof2019.csv"
ENROLL_ROOT   = "./enr_audio/eval"

# ===== Model import =====
sys.path.append("./")
from src.models.ecapa_tdnn import ECAPA_TDNN

MODEL_PATH = "./pretrained/pretrain.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ECAPA_TDNN(C=1024).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
raw_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
filtered = {k.replace("speaker_encoder.", ""): v for k, v in raw_state.items() if k.startswith("speaker_encoder.")}
model.load_state_dict(filtered if len(filtered) > 0 else raw_state, strict=False)
model.eval()

def read_mono_float32(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav

@torch.no_grad()
def extract_embedding(wav_path):
    wav = read_mono_float32(wav_path)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    emb = model(wav_tensor, aug=False).squeeze().detach().cpu().numpy()
    return emb

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

def find_test_file(utt_id):
    for base_dir in [AUDIO_DIR, AUDIO_DIR2]:
        for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
            cand = os.path.join(base_dir, utt_id + ext)
            if os.path.exists(cand):
                return cand
    return None

protocol_entries = []
with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            enroll_id, utt_id, label, trial_type = parts
            protocol_entries.append((enroll_id, utt_id, label, trial_type))
print(f"[INFO] Loaded {len(protocol_entries)} protocol entries")

# ===== Scoring =====
embeddings, enr_embeddings = {}, {}
results, skipped = [], []

for enroll_id, utt_id, label, trial_type in tqdm(protocol_entries, desc="Scoring protocol pairs"):
    test_fp = find_test_file(utt_id)
    if not test_fp:
        skipped.append((utt_id, "Test file not found"))
        continue

    enroll_path = find_enroll_path(enroll_id)
    if not enroll_path or not os.path.exists(enroll_path):
        skipped.append((utt_id, f"Enroll audio not found for {enroll_id}"))
        continue

    try:
        if test_fp not in embeddings:
            embeddings[test_fp] = extract_embedding(test_fp)
        if enroll_path not in enr_embeddings:
            enr_embeddings[enroll_path] = extract_embedding(enroll_path)

        s = cosine(embeddings[test_fp], enr_embeddings[enroll_path])
        results.append((utt_id, enroll_id, s, label, trial_type))
    except Exception as e:
        skipped.append((utt_id, str(e)))

# ===== Save Scores =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,cosine_score,label,trial_type\n")
    for utt_id, enroll_id, s, label, trial_type in results:
        fw.write(f"{utt_id},{enroll_id},{s:.6f},{label},{trial_type}\n")

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} items. First few:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")
