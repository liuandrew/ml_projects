import torch
from torch import nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import os

from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module):
    #In multi-head attention, split the x vector into 8 heads
    #
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        
        # unifies outputs of different heads into a singl k-vector
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        
        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)
        
        #fold head into batch dimension
        #contiguous allows us to use view correctly
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot has size (b * h, t, t)
        dot = F.softmax(dot, dim=2)
        
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        
        return self.unifyheads(out)
        
        
        

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        
        self.attention = SelfAttention(k, heads=heads)
        
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)
    
    

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)
        
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        
        self.toprobs = nn.Linear(k, num_classes)
        
    def forward(self, x):
        '''
        :param x: A (b, t) tensor of integer values representing words
        :return: A (b, c) tensor of log_probabilities over classes
        '''
        
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = torch.arange(t).to(device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        
        x = tokens + positions
        x = self.tblocks(x)
        
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)



def collate_batch(batch):
    label_list, text_list = [], []
    for label, line in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(line), dtype=torch.int64).to(device)
        text_list.append(processed_text)
    
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    text_list = torch.stack(text_list).to(device)
    return label_list.to(device), text_list.to(device)



def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 200
    start_time = time.time()
    
    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch,
                        idx, len(dataloader), total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count




if __name__ == "__main__":
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for label, line in data_iter:
            yield tokenizer(line)
            
    if 'imdb_vocab' not in os.listdir():
        print('Downloading text data...')
        train_iter = IMDB(split='train')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
        print('Building vocab...')
        vocab.set_default_index(vocab['<unk>'])
        torch.save(vocab, 'imdb_vocab')
    else:
        print('Loading vocab...')
        vocab = torch.load('imdb_vocab')

    max_seq_len = 256
    #trim and pad
    def text_pipeline(line):
        line = vocab(tokenizer(line)[:max_seq_len])
        if len(line) < max_seq_len:
            line = line + [0] * (max_seq_len - len(line))
        return line
    # text_pipeline = lambda x:vocab(tokenizer(x)[:max_seq_len])
    label_pipeline = lambda x: 0 if x == 'neg' else 1

    EPOCHS = 3
    LR = 5
    BATCH_SIZE = 64

    model = Transformer(k=64, heads=8, depth=4, seq_length=max_seq_len, num_tokens=len(vocab), num_classes=2).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    print('Downloading text data...')
    train_iter, test_iter = IMDB()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])



    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch)


    print('Beginning training...')
    print('Allocated: {:d}, Reserved: {:d}'.format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
            torch.save(model, 'model')
        
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:8.3f} '.format(epoch,
                                                                time.time() - epoch_start_time, accu_val))
        print('-' * 59)