import torch
import torch.nn.functional as F

class ContrastiveLoss:
    def __init__(self, logit_scale):
        self.logit_scale = logit_scale

    def cosine_sim_with_scale(self, x, y):
        sim = torch.matmul(x, y.T)
        scale = torch.exp(self.logit_scale)

        return sim * scale

    def __call__(self, image_embeds, text_embeds, batch_size):
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        logits_per_image = self.cosine_sim_with_scale(image_embeds, text_embeds)
        logits_per_text = logits_per_image.T

        # batch_size = text_inputs.shape[0]
        labels = torch.arange(batch_size)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i + loss_t) / 2

        return loss
