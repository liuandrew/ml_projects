import torch
from torch import nn, Tensor
from torchtext.datasets import YelpReviewFull
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math
from typing import Tuple
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import copy
import time

class TransformerSentiment(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, num_classes=5, dropout=0.3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoder(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.sentiment_ff = nn.Linear(d_model, num_classes)
        
        # self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.view([-1, 1, self.d_model]), src_mask)
        output = self.sentiment_ff(output)
        output = output.sum(dim=0).squeeze() / math.sqrt(src.size(0))
        output = F.softmax(output, dim=0) #not sure whether to include softmax here
        return output
        
        
        
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




#Prepare data
train = YelpReviewFull(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, map(lambda x: x[1], train)))

vocab.set_default_index(0)

def data_process(text_iter):
    '''
    Process data from YelpReviewFull
    Return scores and tokenized text split into two tensors
    '''
    text_data = []
    score_data = []
    for item in tqdm(text_iter):
        text_data.append(torch.tensor(vocab(tokenizer(item[1])), dtype=torch.long))
        score_data.append(torch.tensor([[item[0]]], dtype=torch.long))
        
    score_data = torch.cat(tuple(filter(lambda t: t.numel() > 0, score_data)))
    score_data = F.one_hot(score_data - 1).squeeze() #convert to one hot for comparison
    
    return score_data, text_data

train_iter, test_iter = YelpReviewFull()
train_data = data_process(train_iter)
test_data = data_process(test_iter)

valid_data = (train_data[0][0:20000], train_data[1][0:20000])
train_data = (train_data[0][20000:], train_data[1][20000:])



#Setup model
ntokens = len(vocab)
emsize = 200 #same as d_model
d_hid = 200 #FF network width
nlayers = 2 #number of attention layers
nhead = 2 #number of heads per attention layer
dropout = 0.2
model = TransformerSentiment(ntokens, emsize, nhead, d_hid, nlayers, dropout=dropout).to(device)



bsz = 32
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
def get_batch(data, i=0):
    return data[0][i:i+bsz], data[1][i:i+bsz]



def train(model):
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data[1]) // bsz

    for batch, i in enumerate(tqdm(range(0, len(train_data[1]) - 1, bsz))):
        targets, inputs = get_batch(train_data, i)

        outputs = []
        for i in range(len(inputs)):
            output = model(inputs[i])
            outputs.append(output)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets.type(torch.FloatTensor))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f' epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches'
                        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                        f'loss {cur_loss:5.2f} | ppl{ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            
            
def evaluate(model, eval_data):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, len(eval_data[1]) - 1, bsz):
            targets, inputs = get_batch(eval_data, i)
            outputs = []
            for i in range(len(inputs)):
                output = model(inputs[i])
                outputs.append(output)
            outputs = torch.stack(outputs)
            
            total_loss += criterion(outputs, targets)
    return total_loss / len(eval_data[1])
            



best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, valid_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-'*89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s |'
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-'*89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        torch.save(model, 'best_model.pt')

    scheduler.step()