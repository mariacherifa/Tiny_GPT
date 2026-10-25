########################################################################
#                       Attention classes 
########################################################################
import torch 
import torch.nn as nn 

class SingleAttention(nn.Module):
    def __init__(self, n_embd, head_size, block_size,  dropout):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias = True)
        self.key = nn.Linear(n_embd, head_size, bias= True )
        self.value = nn.Linear(n_embd, head_size, bias = True)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x): 

        B,T,C = x.shape 

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)

        wei  = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        

        wei  = torch.softmax(wei, dim=-1)

        wei = self.dropout(wei)


        output = wei @ v 

        return output 


class MultiHeadAttention(nn.Module): 
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        assert n_embd % num_heads == 0, \
           "n_embd must be divisible by num_heads" 

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([SingleAttention(n_embd, head_size, block_size,  dropout) for _ in range(num_heads)])

        self.proj = nn.Linear(num_heads * head_size, n_embd)

        self.dropout = nn.Dropout(dropout)


    def forward(self,x): 

        out = torch.cat([head(x) for head in self.heads],  dim =-1)

        out = self.proj(out)

        out = self.dropout(out)

        return out 




 




    

