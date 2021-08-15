import torch
class Free:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def get_backup(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()

    def attack(self, epsilon=1.0, emb_name='embedding'):
        adv_param = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    if torch.norm(r_at) > epsilon:
                        r_at = epsilon * r_at / torch.norm(r_at)
                    param.data.add_(r_at)
                    adv_param[emb_name] = param.data.clone()
        return adv_param

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def restore_adv(self, data, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = data