import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl

class GeMSEncoder(nn.Module):
    def __init__(self, embeddings: nn.Module, latent_dim: int, slate_size: int, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim

        self.embeddings = embeddings
        self.item_embedding_dim = embeddings.weight.shape[1]
        self.slate_size = slate_size
        self.model = nn.Sequential(
            nn.Linear((self.item_embedding_dim + 1) * self.slate_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2),
        )

    def forward(self, slate, clicks):
        # Convert slate to LongTensor if necessary
        slate = slate.int()

        # Convert clicks to float32 if not already
        clicks = clicks.float()

        # Ensure embeddings weights are float32
        self.embeddings.weight.data = self.embeddings.weight.data.float()

        enc = self.model(torch.cat([self.embeddings(slate).flatten(start_dim = -2), clicks], dim = -1))
        return enc[:, :self.latent_dim], enc[:, self.latent_dim:]

class GeMSDecoder(nn.Module):
    def __init__(self, embeddings: nn.Module, latent_dim: int, slate_size: int, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.embeddings = embeddings
        self.item_embedding_dim = embeddings.weight.shape[1]
        self.slate_size = slate_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, (self.item_embedding_dim +1) * self.slate_size),
        )

    def forward(self, proto_action):
        logits = self.model(proto_action).reshape(-1, self.slate_size, self.item_embedding_dim +1)
        # Convert logits to the same dtype as embeddings.weight
        logits = logits.to(self.embeddings.weight.dtype)
        item_logits = logits[:, :, :-1] @ self.embeddings.weight.detach().t()
        click_logits = logits[:, :, -1]
        return item_logits, click_logits

class GeMS(pl.LightningModule):
    def __init__(self, item_embeddings: str, data_dir: str, dataset: str, latent_dim: int, 
                    gems_lr: float, lambda_KL: float, lambda_click: float, slate_size: int, **kwargs):
        super().__init__()
        weights = torch.tensor(np.load(os.path.join("sardine", "embeddings", f"item_embeddings_numitems{kwargs['num_items']}.npy")))      
        self.embeddings = nn.Embedding(*weights.shape, _weight = weights).requires_grad_(False)
        self.item_embedding_dim = weights.shape[1]
        self.slate_size = slate_size
        self.encoder = GeMSEncoder(embeddings = self.embeddings, latent_dim = latent_dim, slate_size=self.slate_size, **kwargs)
        self.decoder = GeMSDecoder(embeddings = self.embeddings, latent_dim = latent_dim, slate_size=self.slate_size, **kwargs)

        self.lr = gems_lr
        self.lambda_click = lambda_click
        self.lambda_KL = lambda_KL

        self.validation_step_outputs = []
    
    def forward(self, x):
        latent_mean, _ = self.encoder(x)
        return self.decoder(latent_mean).argmax(dim = -1)

    def training_step(self, batch, batch_idx):
        slate, clicks = batch.actions, batch.observations["clicks"].float()
        latent_mean, log_latent_var = self.encoder(slate, clicks)

        latent_var = log_latent_var.exp()
        latent_sigma = torch.sqrt(latent_var)
        latent_sample = latent_mean + torch.randn_like(latent_sigma) * latent_sigma

        item_logits, click_logits = self.decoder(latent_sample)
        slate_loss = nn.CrossEntropyLoss(reduction = 'mean')(item_logits.transpose(1,2), slate)
        click_loss = nn.BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks)   

        prior_mean, log_prior_var = torch.zeros_like(latent_mean), torch.zeros_like(latent_mean)
        prior_var = log_prior_var.exp()

        mean_term = ((latent_mean - prior_mean) ** 2) / prior_var
        KL_loss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()

        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KL_loss

        self.log("train/loss", loss, prog_bar=True, logger=True)
        self.log("train/slate_loss", slate_loss, logger=True)
        self.log("train/click_loss", slate_loss, logger=True)
        self.log("train/KL_loss", KL_loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        slate, clicks = batch.actions, batch.observations["clicks"].float()
        latent_mean, log_latent_var = self.encoder(slate, clicks)

        latent_var = log_latent_var.exp()

        item_logits, click_logits = self.decoder(latent_mean)
        slate_loss = nn.CrossEntropyLoss(reduction = 'mean')(item_logits.transpose(1,2), slate)
        click_loss = nn.BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks)   

        prior_mean, log_prior_var = torch.zeros_like(latent_mean), torch.zeros_like(latent_mean)
        prior_var = log_prior_var.exp()

        mean_term = ((latent_mean - prior_mean) ** 2) / prior_var
        KL_loss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()
        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KL_loss   

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/slate_loss", slate_loss, on_epoch=True, logger=True)
        self.log("val/click_loss", slate_loss, on_epoch=True, logger=True)
        self.log("val/KL_loss", KL_loss, on_epoch=True, logger=True)

        self.validation_step_outputs.append(torch.tensor([loss, slate_loss, click_loss]))
        return loss
    
    def on_validation_epoch_end(self) -> None:
        ## Print metrics
        if self.global_rank == 0:
            loss = torch.stack(self.validation_step_outputs, dim = 0).mean(dim = 0)
            print(f"Epoch {self.current_epoch} : val_loss = {loss[0]:.2f}, slate_loss = {loss[1]:.2f}, click_loss = {loss[2]:.2f}")
        self.validation_step_outputs.clear() 
        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)





