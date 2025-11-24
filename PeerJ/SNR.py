import os
import glob
import math
import soundfile as sf
import numpy as np
from tqdm import tqdm

ADV_DIR = "./PeerJ/exp1/CM_aasist/cm_pgd_0007"  
ORIG_DIR = "./LA/ASVspoof2019_LA_eval/flac"         

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

def load_audio_any(path):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav, sr

def match_and_resample(wav, sr_src, sr_tgt):
    if sr_src == sr_tgt:
        return wav, sr_src
    if not HAVE_LIBROSA:
        raise RuntimeError(f"src={sr_src}, tgt={sr_tgt}")
    wav_rs = librosa.resample(wav, orig_sr=sr_src, target_sr=sr_tgt, res_type="kaiser_best")
    return wav_rs.astype(np.float32), sr_tgt

def align_lengths(x, y):
    n = min(len(x), len(y))
    return x[:n], y[:n]

def snr_db(ref, test, eps=1e-12):
    noise = test - ref
    p_sig = float(np.sum(ref.astype(np.float64)**2))
    p_nse = float(np.sum(noise.astype(np.float64)**2))
    return 10.0 * math.log10((p_sig + eps) / (p_nse + eps))

def main():
    orig_index = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(ORIG_DIR, "**", "*.*"), recursive=True)
    }
    adv_files = sorted(glob.glob(os.path.join(ADV_DIR, "**", "*.*"), recursive=True))

    missing = 0
    sr_mismatch_skip = 0
    snr_values = []

    for adv_path in tqdm(adv_files, desc="Computing SNR"):
        adv_stem_full = os.path.splitext(os.path.basename(adv_path))[0]

        if "_LA_" in adv_stem_full:
            stem = adv_stem_full.split("_LA_")[0]   # "LA_E_1008476_LA_0001" → "LA_E_1008476"
        else:
            stem = adv_stem_full

        if stem not in orig_index:
            missing += 1
            continue

        orig_path = orig_index[stem]

        try:
            adv_wav, adv_sr = load_audio_any(adv_path)
            orig_wav, orig_sr = load_audio_any(orig_path)

            try:
                adv_wav, adv_sr = match_and_resample(adv_wav, adv_sr, orig_sr)
            except RuntimeError:
                sr_mismatch_skip += 1
                continue

            orig_wav, adv_wav = align_lengths(orig_wav, adv_wav)
            if len(orig_wav) == 0:
                continue

            snr = snr_db(orig_wav, adv_wav)
            snr_values.append(snr)
            print(f"{stem}: SNR = {snr:.4f} dB")

        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    # 평균 SNR 계산
    if snr_values:
        avg_snr = np.mean(snr_values)
        print(f"\nMean SNR = {avg_snr:.4f} dB")
    else:
        print("\nERROR")
        
if __name__ == "__main__":
    main()
