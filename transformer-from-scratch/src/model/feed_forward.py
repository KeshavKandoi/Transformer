"""A shared two-layer ReLU network applied independently at each position."""

from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(d_ff) is not int or d_ff < 1:
            raise ValueError("d_ff must be a positive integer")
        if type(dropout) not in (int, float) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be a number in [0, 1]")
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or x.size(-1) != self.linear1.in_features:
            raise ValueError(f"FFN expects a last dimension of size {self.linear1.in_features}")
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
