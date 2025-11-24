import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from easydict import EasyDict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.conformer_tcm import Model as Conformer

# ===== Path =====
MODEL_PATH = "./best_4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR1 = "./LA/ASVspoof2019_LA_eval/flac/"
AUDIO_DIR2 = "./PeerJ/exp1/CM_aasist-ssl3/cm_bim_0007"  # sub
PROTOCOL_FILE = "./PeerJ/protocol/ASVspoof2019_cmatt.csv"
OUTPUT_TXT = "./PeerJ/exp1/CM_aasist-ssl/conformer-tcm.txt"

# ===== model args =====
args = EasyDict({
    "emb_size": 144,
    "num_encoders": 4,
    "heads": 4,
    "kernel_size": 31
})

model = Conformer(args=args, device=DEVICE).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model.eval()

def find_audio_path(utt_id):
    for base_dir in [AUDIO_DIR1, AUDIO_DIR2]:
        for ext in [".flac", ".wav"]:
            path = os.path.join(base_dir, utt_id + ext)
            if os.path.exists(path):
                return path
    return None

scores, labels = [], []
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")

    with open(PROTOCOL_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scoring Conformer-TCM"):
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        enroll_id, utt_id, label, trial_type = parts[0], parts[1], parts[2], parts[3]
        audio_path = find_audio_path(utt_id)
        if not audio_path:
            print(f"Missing audio: {utt_id}")
            continue

        wav, sr = sf.read(audio_path)
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = model(wav_tensor)
            cm_score = logits[0, 1].cpu().item()

        scores.append(cm_score)
        labels.append(1 if label == "bonafide" else 0)

        fw.write(f"{utt_id},{enroll_id},{label},{trial_type},{cm_score:.6f}\n")
        fw.flush()

# ===== EER  =====
if len(labels) == 0 or len(scores) == 0:
    raise RuntimeError("❌ No valid scores or labels found. Check protocol format.")

fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"\nConformer-TCM EER: {eer * 100:.2f}%")
print(f"Threshold at EER: {eer_threshold:.4f}")
print(f"All scores saved to: {OUTPUT_TXT}")
