import torch
import time

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.Wk = torch.nn.Linear(embed_size, embed_size)
        self.Wq = torch.nn.Linear(embed_size, embed_size)
        self.Wv = torch.nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
    
    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        
        # Compute queries, keys, values
        Q = self.Wq(x)  # (batch_size, seq_len, embed_size)
        K = self.Wk(x)  # (batch_size, seq_len, embed_size)
        V = self.Wv(x)  # (batch_size, seq_len, embed_size)

        # Compute attention scores
        scores = Q @ K.transpose(1, 2) / (self.embed_size ** 0.5)  # (batch_size, seq_len, seq_len)
        inf_mask = torch.full_like(scores, float('-inf'))
        mask = torch.triu(inf_mask, diagonal=1)
        scores = scores + mask
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        out = attn_weights @ V  # (batch_size, seq_len, embed_size)

        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.Wk = torch.nn.Linear(embed_size, embed_size)
        self.Wq = torch.nn.Linear(embed_size, embed_size)
        self.Wv = torch.nn.Linear(embed_size, embed_size)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()

        # Compute queries, keys, values
        Q = self.Wq(x)  # (batch_size, seq_len, embed_size)
        K = self.Wk(x)  # (batch_size, seq_len, embed_size)
        V = self.Wv(x)  # (batch_size, seq_len, embed_size)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        inf_mask = torch.full_like(scores, float('-inf'))
        mask = torch.triu(inf_mask, diagonal=1)
        scores = scores + mask
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        out = attn_weights @ V  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
        out = self.fc_out(out)  # (batch_size, seq_len, embed_size)
    
        return out

class Transfomer(torch.nn.Module):
    def __init__(self, embed_size, ff_hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads=num_heads)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_size, ff_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_size, embed_size)
        )
        self.norm2 = torch.nn.LayerNorm(embed_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.attention(self.norm1(x))
        x += res
        res = x
        x = self.dropout(self.ffn(self.norm2(x)))
        x += res
        return x
    
class Model(torch.nn.Module):
    def __init__(self, vocab_size, seq_len, embed_size, ff_hidden_size, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = torch.nn.Embedding(seq_len, embed_size)
        self.transformer_layers = torch.nn.ModuleList([
            Transfomer(embed_size, ff_hidden_size, num_heads, dropout) for _ in range(num_layers)
            ])
        self.ffn_out = torch.nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(x.device)

        x = self.embedding(x) + self.positional_embedding(positions)

        for layer in self.transformer_layers:
            x = layer(x)
        
        logits = self.ffn_out(x)

        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0):
        self.eval()
        generated = prompt.clone()
        time_start = time.time()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat((generated, next_token), dim=1)
        time_end = time.time()
        return generated, time_end - time_start
