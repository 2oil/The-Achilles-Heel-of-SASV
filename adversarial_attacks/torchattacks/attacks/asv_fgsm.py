import os
import torch
import torch.nn.functional as F
import librosa
import glob
import random
import numpy as np

class ASV_FGSM:
    def __init__(self, asv_model, enroll_path, input_path, device, eps=0.007, alpha=0.5, seed=None):
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.asv_model = asv_model
        self.enroll_path = enroll_path
        self.input_path = input_path
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"Seed fixed to {self.seed}")

    def _load_enrollment_embedding(self, spk_id):
        enroll_candidates = glob.glob(os.path.join(self.enroll_path, f"{spk_id}_*.flac"))
        if len(enroll_candidates) == 0:
            print(f"[WARNING] No enrollment audio found for speaker {spk_id}")
            return None

        enroll_audio_path = enroll_candidates[0]
        audio, _ = librosa.load(enroll_audio_path, sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.asv_model(audio_tensor).squeeze(0).cpu()

        return embedding

    def forward(self, target_spk=None, test_utterance=None):
        target_embedding = self._load_enrollment_embedding(target_spk)
        if target_embedding is None:
            return None

        test_path = os.path.join(self.input_path, f"{test_utterance}.flac")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}")
            return None

        audio, _ = librosa.load(test_path, sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        audio_tensor.requires_grad = True

        original_embedding = self.asv_model(audio_tensor).squeeze(0)
        asv_before_score = F.cosine_similarity(target_embedding.to(self.device), original_embedding, dim=0).detach().cpu().numpy()

        asv_loss = F.cosine_similarity(target_embedding.to(self.device), original_embedding, dim=0).mean()
        grad = torch.autograd.grad(asv_loss, audio_tensor)[0]
        adv_audio = torch.clamp(audio_tensor + self.eps * grad.sign(), -1, 1)

        with torch.no_grad():
            attacked_embedding = self.asv_model(adv_audio).squeeze(0)
            after_cosine_sim = F.cosine_similarity(target_embedding.to(self.device), attacked_embedding, dim=0).detach().cpu().numpy()

        return asv_before_score, after_cosine_sim, adv_audio.squeeze(0).detach().cpu().numpy()

