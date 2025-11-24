import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack
import random

class CM_PGD(Attack):
    def __init__(self, model, device, eps=0.007, alpha=0.001, steps=10, seed=None):
        super().__init__("PGD", model)
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

        # random start
        images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
        images = torch.clamp(images, min=-1, max=1)

        self.model.eval()
        with torch.no_grad():
            before_attack = self.model(ori_images)
            before_score = before_attack[:, 1].cpu().numpy().ravel()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

            images = images + self.alpha * grad.sign()
            eta = torch.clamp(images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=-1, max=1).detach()

        with torch.no_grad():
            after_attack = self.model(images)
            after_score = after_attack[:, 1].cpu().numpy().ravel()

        return before_score, after_score, images
