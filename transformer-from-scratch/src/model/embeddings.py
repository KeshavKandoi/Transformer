"""Compose scaled token lookup, positions, dropout and padding."""

from torch import Tensor, nn

from .positional_encoding import SinusoidalPositionalEncoding
from .token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float, pad_id: int
    ) -> None:
        super().__init__()
        if type(dropout) not in (int, float) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be a number in [0, 1]")
        self.pad_id = pad_id
        self.position_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        embeddings = self.token_embedding(token_ids)
        positioned = self.position_encoding(embeddings)
        return self.dropout(positioned).masked_fill(token_ids.eq(self.pad_id).unsqueeze(-1), 0.0)
