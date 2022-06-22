import torch.nn as nn
import torch
from torch import Tensor
from numpy import sin, cos

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        device = "cuda:0",
    ):
        super().__init__()

        self.model = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=0,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        ).float().to(device)

        # remove encoder
        self.model.encoder = nn.Identity()
        self.model.encoder.forward = lambda *args, **kwargs: args[0]

        self.embedding = nn.Embedding(vocab_len, d_model)
        self.position_encoding = self._position_encoding(max_context_len, d_model).to(device)
        self.src_mask = self.generate_square_subsequent_mask(max_context_len).to(device)

        self.decoding = nn.Linear(d_model, vocab_len)

        self.num_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.num_params += sum([p.numel() for p in self.embedding.parameters() if p.requires_grad])
          
    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore
        embedded = self.embedding(indices)
        return pe + embedded
    
    @staticmethod
    def _position_encoding(context_len: int, d_model: int):
        rows = [
            torch.tensor([
                sin(pos / (10000 ** (i / d_model)))
                if i % 2 == 0
                else cos(pos / (10000 ** ((i - 1) / d_model)))
                for i in range(d_model)
            ])
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)
        return stack.T
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, indices: Tensor) -> Tensor:
        embedded = self.embed(indices).float()
        mask = self.src_mask[:indices.shape[-1], :indices.shape[-1]].float()
        # model forward takes src, tgt, src_mask, tgt_mask
        return self.decoding(self.model(torch.zeros_like(embedded), embedded, mask, mask))
