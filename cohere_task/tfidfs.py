from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def tfidfs(tokenizer, lines, df_shift=0):
    '''
    With a given tokenizer, calculate TF-IDF of set of documents (lines)
    Optionally pass a df_shift to aritificially inflate df count values
    and truncate words that are too common from tf_idfs
    (Also nonlinearly shifts tf-idf scores for other entries)
    '''
    tfs = np.zeros(tokenizer.get_vocab_size())
    
    all_tfs = []
    for line in lines:
        encoded = tokenizer.encode(line).ids
        tfs = np.zeros(tokenizer.get_vocab_size())
        for ids in encoded:
            tfs[ids] += 1
        all_tfs.append(tfs / len(encoded))
        
    tfs_df = pd.DataFrame(all_tfs, columns=range(tokenizer.get_vocab_size()))
    
    df_counts = (((tfs_df != 0) * 1).sum(axis=0))
    df_counts = df_counts + df_shift
    idfs = len(tfs_df) / df_counts
    idfs = np.log(idfs)

    tf_idfs = tfs_df * idfs

    return tfs_df, idfs, tf_idfs



def train_tokenizer(lines):
    '''
    Train a basic word-level tokenizer for tf-idf
    '''
    tokenizer = Tokenizer(WordLevel(unk_token='<unk>'))

    trainer = WordLevelTrainer(special_tokens=['<unk>', '<reas>', '<drug>', '<pair>',
                                               '<sep>', '<mask>', '<s>', '</s>'])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(lines, trainer=trainer)
    
    return tokenizer




def main():

    patient_df = pd.read_csv('patient_record_summary.csv')
    patient_df = patient_df.fillna('')
    lines = list(patient_df['chief_complaint'].str.lower() + ' \n ' + patient_df['present_history'].str.lower())

    # train a tokenizer for tf-idf analysis            
    tokenizer = train_tokenizer(lines)


    tfs_df, idfs, tf_idfs = tfidfs(tokenizer, lines)
    plt.figure(figsize=(8,8))
    plt.title('Number of documents word is found in')

    plt.scatter(range(tokenizer.get_vocab_size()), (((tfs_df != 0) * 1).sum(axis=0)).sort_values(ascending=False))
    plt.plot([0, tfs_df.shape[1]], [150, 150], '--')

    plt.ylabel('# of documents found word is found in')
    plt.xlabel('Rank')
    plt.savefig('word_frequencies.png', bbox_inches='tight')
    plt.show()


    tfs_df, idfs_shifted, tf_idfs_shifted = tfidfs(tokenizer, lines, 150)

    top_500 = tf_idfs_shifted.sum(axis=0).sort_values(ascending=False).head(500)
    print('\n Top 500 words in total tf-idf score from Chief Complaint and History after DFs shift \n')
    print(tokenizer.decode(np.array(top_500.index)))



if __name__ == '__main__':
    main()