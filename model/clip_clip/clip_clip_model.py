import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel
from model.constrastive_loss import ContrastiveLoss

class ClipCLIP(nn.Module):
    def __init__(self, load_path=None, device="cpu"):
        super().__init__()

        self.device = torch.device(device)

        if load_path is not None:
            self.clip_model = torch.load(load_path, map_location=self.device)
        else:
            # Load CLIP model from Hugging Face
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)

    @classmethod
    def from_save(cls, load_path, device="cpu"):
        return cls(load_path=load_path, device=device)

    def infer(self, inputs, ret_key="logits_per_image", ret_np=False):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to device
        outputs = self.clip_model(**inputs)

        ret = outputs if ret_key is None else outputs[ret_key]

        return ret.detach().cpu().numpy() if ret_np else ret

    def forward(self, inputs, ret_key="logits_per_image", ret_np=False):
        return self.infer(inputs, ret_key=ret_key, ret_np=ret_np)

    def prepare_training(self, initial_lr, decay_steps, decay_factor, momentum, weight_decay,):
        self.loss_fn = ContrastiveLoss(self.clip_model.logit_scale)

        self.optimizer = optim.SGD(self.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_schedule = optim.lr_scheduler.StepLR(self.optimizer, decay_steps, decay_factor)

    def custom_fit(self, train_loader, initial_epoch, final_epoch, callbacks, effective_batch_size):
        self.train()
        callbacks.on_train_begin()

        for epoch in range(initial_epoch, final_epoch):
            callbacks.on_epoch_begin(epoch)

            accumulated_embeds = [[], []]  # For images and texts
            self.optimizer.zero_grad()

            for step, x_batch in enumerate(train_loader):
                # callbacks.on_train_batch_begin(step)

                x_batch = {k: v.to(self.device) for k, v in x_batch.items()}  # could be in collate_fn
                predictions = self(x_batch, ret_key=None)

                accumulated_embeds[0].append(predictions["image_embeds"])
                accumulated_embeds[1].append(predictions["text_embeds"])

                # callbacks.on_train_batch_end(step, logs={"loss": -1})

                # Gradient accumulation based on effective batch size
                used_batch_size = x_batch["pixel_values"].shape[0]
                if len(accumulated_embeds[0]) >= effective_batch_size // used_batch_size:
                    image_embeds = torch.cat(accumulated_embeds[0], dim=0)
                    text_embeds = torch.cat(accumulated_embeds[1], dim=0)
                    accumulated_embeds = [[], []]  # Reset accumulation buffer

                    # sim = self.cosine_sim_with_scale(image_embeds, text_embeds)
                    loss = self.loss_fn(image_embeds, text_embeds, effective_batch_size)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_schedule.step()

                    callbacks.on_epoch_end(epoch, logs={"loss": loss.item()})

                    break

        callbacks.on_train_end()
