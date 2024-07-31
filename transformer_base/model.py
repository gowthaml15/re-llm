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
    def __init__(self, eps:float=10**-5)->None:
        super().__init__()
        self.eps = eps
        # Below 2 values will be learned during the training processing to adjust the layer normalization
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,heads:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        assert d_model % heads == 0, "d_model is not divisible by heads"

        self.d_k = d_model//heads
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Liner(d_model,d_model)
    
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        # (batch, head,seq_len,d_k) * (batch, head,d_k,seq_len) -> (batch, head, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Scaled Dot-Product Attention, attention scores
        # (batch, head, seq_len, seq_len) * (batch, head,seq_len,d_k) -> (batch, head, seq_len, d_k), (batch, head,seq_len,seq_len)
        return (attention_scores@value), attention_scores 

        

    def forward(self, q, k, v, mask): 
        query = self.w_q(q) # (batch,seq_len,d_model) -> (batch,seq_len,d_model)
        key = self.w_k(k) # (batch,seq_len,d_model) -> (batch,seq_len,d_model)
        values = self.w_v(v) # (batch,seq_len,d_model) -> (batch,seq_len,d_model)
        #(batch,seq_len,d_model)-->(batch,seq_len,head,d_k)-->(batch,head,seq_len,d_k) 
        query = query.view(query.shape[0],query.shape[1],self.heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.heads, self.d_k).transpose(1,2)
        values = values.view(values.shape[0],values.shape[1],self.heads, self.d_k).transpose(1,2)

        #attention_scores are for visualization the score mattrix for that sentence
        x,self.attention_scores = MultiHeadAttention.attention(query,key,values,mask,self.dropout) # (batch, head, seq_len, d_model), (batch, head,seq_len,seq_len)

        #(batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k) -> (btach, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],self.heads*self.d_model)

        #(batch,seq_len, d_model) -> (batch,seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNorm()
    def forward(self,x, sublayer:FeedForward):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,attention_block:MultiHeadAttention, feed_forward:FeedForward, dropout:float)->None:
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.residual_block = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        x = self.residual_block[0](x,lambda x:self.attention_block(x,x,x,src_mask))
        x = self.residual_block[1](x,lambda x:self.feed_forward(x))
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self,x, mask):
        x = self.layers(x,mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, attention_blocks:MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward:FeedForward, dropout:float) -> None:
        super().__init__()
        self.attention_block = attention_blocks
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        self.dropout = dropout

        self.residual_blocks = nn.ModuleDict([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): #src_mask from encoder and tgt_mask from decoder
        x = self.residual_blocks[0](x, lambda x:self.attention_block(x,x,x,tgt_mask))
        x = self.residual_blocks[1](x, lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        




