#####################################################################################
#                               Transformer block 
#####################################################################################

import torch 
import torch.nn as nn 
from src.attention import MultiHeadAttention

class MLP(nn.Module): 
    def __init__(self, n_embd, dropout):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd, bias = True), 
                                 nn.ReLU(), 
                                 nn.Linear(4*n_embd, n_embd, bias = True))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        return self.dropout(self.mlp(x))


class BlockTransformer(nn.Module): 
    def __init__(self, n_embd, num_heads, block_size, dropout ):
        super().__init__()

        self.att = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): 

        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x 

