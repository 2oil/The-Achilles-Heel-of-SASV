import os
import sys
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ===== import RawNet2 =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.raw_net2 import RawNet  # DF-RawNet2

# ===== Path =====
MODEL_PATH   = "./pretrained_rawnet2/pre_trained_DF_RawNet2.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR    = "./LA/ASVspoof2019_LA_eval/flac"
AUDIO_DIR2   = "./PeerJ/exp1/CM_aasist3/cm_bim_0007"
PROTO_FILE   = "./PeerJ/protocol/ASVspoof2019.csv"
OUTPUT_TXT   = "./PeerJ/exp1/CM_aasist/rawnet2.txt"

# ===== RawNet2 settings =====
model_config = {
    "nb_samp": 64600, "first_conv": 1024, "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024,
    "nb_gru_layer": 3, "nb_classes": 2
}

# ===== Load Model =====
model = RawNet(d_args=model_config, device=DEVICE).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()

# ===== utils =====
MAX_LEN = 64600
def load_and_pad(path):
    wav, sr = sf.read(path)
    if len(wav) > MAX_LEN:
        wav = wav[:MAX_LEN]
    elif len(wav) < MAX_LEN:
        wav = np.tile(wav, (MAX_LEN // len(wav)) + 1)[:MAX_LEN]
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def find_audio(utt_id):
    for base_dir in [AUDIO_DIR, AUDIO_DIR2]:
        for ext in [".flac", ".wav"]:
            path = os.path.join(base_dir, utt_id + ext)
            if os.path.exists(path):
                return path
    return None

# ===== protocol =====
entries = []
with open(PROTO_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            enroll_id, utt_id, label, trial_type = parts
            entries.append((enroll_id, utt_id, label, trial_type))
print(f"[INFO] Loaded {len(entries)} protocol entries")

# ===== Scoring =====
results, skipped = [], []
for enroll_id, utt_id, label, trial_type in tqdm(entries, desc="Scoring CM"):
    path = find_audio(utt_id)
    if not path:
        skipped.append((utt_id, "File not found"))
        continue

    try:
        wav_tensor = load_and_pad(path)
        with torch.no_grad():
            logits = model(wav_tensor)
            cm_score = float(logits[0, 1].cpu().numpy())  # bonafide logit

        results.append((utt_id, enroll_id, label, trial_type, cm_score))
    except Exception as e:
        skipped.append((utt_id, str(e)))

# ===== Save results =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")
    for utt_id, enroll_id, label, trial_type, cm_score in results:
        fw.write(f"{utt_id},{enroll_id},{label},{trial_type},{cm_score:.6f}\n")

print(f"\n Done. Saved {len(results)} CM scores to: {OUTPUT_TXT}")
if skipped:
    print(f" Skipped {len(skipped)} items. Example:")
    for p, msg in skipped[:5]:
        print(f" - {p} :: {msg}")
