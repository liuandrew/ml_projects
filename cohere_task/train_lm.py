
# convert chief complaint and present history to text files
# to use as transformers dataset
patient_df = pd.read_csv('patient_record_summary.csv')
patient_df = patient_df.fillna('')
lines = list(patient_df['chief_complaint'].str.lower() + ' \n ' + patient_df['present_history'].str.lower())
file =  open('chief_history.txt', 'w')
for line in texts:
    file.write(line + '\n')
file.close()

# use pretrained tokenizer and model from 'prajjwal1/bert-tiny' and 
# update with new tokens
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

vocab_finder = Tokenizer(WordLevel(unk_token='<unk>'))
trainer = WordLevelTrainer()
vocab_finder.pre_tokenizer = Whitespace()
vocab_finder.train_from_iterator(texts)

tokens = [key for key, value in vocab_finder.get_vocab().items()]

from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = 'prajjwal1/bert-tiny'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
num_added_toks = tokenizer.add_tokens(tokens)

model = AutoModelForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


# Load dataset from txt file
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="chief_history.txt",
    block_size=512
)

# Load mask data collator for LM training
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# load trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./text_model",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=8,
    save_steps=50,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# train and save model
trainer.train()

tokenizer.save_pretrained('text_tokenizer')
trainer.save_model('text_model')


# convert lines from each patient into vectors encoded by model
from transformers import AutoModel

device = torch.device('cpu')
model = AutoModel.from_pretrained('text_model')

def padding_collator(dataset, padding=128):
    inps = []
    for data in dataset:
        padded = torch.zeros(padding)
        inp = data['input_ids']
        inp = torch.tensor(inp)
        
        if len(inp) > len(padded):
            padded[:padding] = inp[:padding]
        else:
            padded[:len(inp)] = inp[:]
        inps.append(padded)
    inps = torch.vstack(inps).to(device).to(torch.int64)
    return inps