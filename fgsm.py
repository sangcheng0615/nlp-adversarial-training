import torch
class FGSM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.01, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                sign = torch.sign(param.grad)
                r_at = epsilon * sign
                param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}