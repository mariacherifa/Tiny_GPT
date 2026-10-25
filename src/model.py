#######################################################################################
#                                 TinyGPT 
#######################################################################################

import torch 
import torch.nn as nn 
from src.transformer import BlockTransformer

class TinyGPT(nn.Module): 
    def __init__(self, n_embd, vocab_size, num_heads, block_size,num_layers, dropout):
        super().__init__()

        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)
        self.blocks_size = block_size

        self.blocks = nn.Sequential(*[BlockTransformer(n_embd, num_heads, block_size, dropout) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)


    def forward(self, ids, targets = None): 
        B, T = ids.shape

        tok_emb = self.token_embd(ids)
        pos = torch.arange(T, device=ids.device)
        pos_emd = self.pos_embd(pos)

        x = tok_emb + pos_emd

        x = self.blocks(x)

        x = self.ln(x)
        
        logits = self.head(x)

        if targets is None: 

            loss = None 

        else: 
            B,T,C = logits.shape

            logits_flat = logits.reshape(B*T,C)
            targets_flat = targets.reshape(B*T)

            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return loss, logits

    def generate(self,idx, max_new_tokens): 

        for _ in range(max_new_tokens): 
            context_crop = idx[:, -self.blocks_size:]

            _, logits = self(context_crop)

            temperature = 0.7 

            logits = logits[:,-1,:]
            logits = logits / temperature

            probs = nn.functional.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1)

            idx = torch.cat((idx,idx_next), dim=1)

        return idx 












