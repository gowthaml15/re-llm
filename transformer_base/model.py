from imports import * 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_siz:int):
        super().__init__()
        self.d_model = d_model # embedding dimension. In paper they used 512
        self.vocab_size = vocab_siz #possible tokens in the input 
        self.embedding = nn.Embedding(vocab_siz,d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # pre-softmax linear trasnformation(look at the transformer paper)
    
class PossitionalEncodding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        positions = torch.arange(0,seq_len,dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        # Positional Encodings based on the formula in paper
        pe[:,0::2] = torch.sin(positions*div_term)
        pe[:,1::2] = torch.cos(positions*div_term)
        # Adding one-more dimension in positional encoding (seq_len,d_model) -> (1,seq_len,d_model)
        pe = pe.unsqueeze(0)
        # Adding the positional encoding to the self.registor_buffer()
        self.register_buffer('pe',pe)
    def forward(self,x):
        # We are adding positional encoddings to the input vectors
        x = x + self.pe[:x.shape[1],:].requires_grad(False) # Disabling the learning by giving the requires grad by false
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, eps:float=10**-05)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Adding
    
    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * ((x-mean)/(std+self.eps)) + self.bias

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = dropout
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.linear_2(x)

        



