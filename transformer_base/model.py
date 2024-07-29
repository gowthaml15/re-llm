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
        #
        pe[:,0::2] = torch.sin(positions*div_term)
        pe[:,1::2] = torch.cos(positions*div_term)



