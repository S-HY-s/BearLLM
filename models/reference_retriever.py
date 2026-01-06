import torch
from torch import nn
from torch.nn import functional as F


class ReferenceRetriever(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, signals):
        if signals.dim() == 2:
            signals = signals.unsqueeze(1)
        zeros = torch.zeros_like(signals)
        paired = torch.cat([signals, zeros], dim=1)
        features = self.encoder(paired)
        pooled = features.mean(dim=2)
        return F.normalize(pooled, dim=1)

    @torch.no_grad()
    def retrieve(self, query_vib, ref_pool_vib):
        if query_vib.dim() == 1:
            query_vib = query_vib.unsqueeze(0)
        if ref_pool_vib.dim() == 1:
            ref_pool_vib = ref_pool_vib.unsqueeze(0)
        if ref_pool_vib.numel() == 0:
            return query_vib.squeeze(0)
        z_query = self.encode(query_vib)
        z_ref = self.encode(ref_pool_vib)
        sim = torch.matmul(z_ref, z_query.T).squeeze(1)
        idx = int(sim.argmax().item())
        return ref_pool_vib[idx]
