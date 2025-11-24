import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack
import random

class CM_BIM(Attack):
    def __init__(self, model, device, eps=0.007, alpha=0.001, steps=10, seed=None):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self._supported_mode = ['default', 'targeted']
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

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        ori_images = images.clone().detach()

        # evaluate original scores before attack
        self.model.eval()
        with torch.no_grad():
            before_logits = self.model(images)
            before_score = before_logits[:, 1].cpu().numpy().ravel()

        # attack loop
        for _ in range(self.steps):
            images.requires_grad = True
            logits = self.model(images)  # [B, 2]
            loss = nn.CrossEntropyLoss()(logits, labels)
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            images = images + self.alpha * grad.sign()
            eta = torch.clamp(images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=-1, max=1).detach()

        # evaluate after attack
        with torch.no_grad():
            after_logits = self.model(images)
            after_score = after_logits[:, 1].cpu().numpy().ravel()

        return before_score, after_score, images
