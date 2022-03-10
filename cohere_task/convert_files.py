import os
import numpy as np
import pandas as pd
from tqdm import tqdm

folder = 'training_20180910/'

txts = [file for file in os.listdir(folder) if '.txt' in file]
anns = [file for file in os.listdir(folder) if '.ann' in file]
ids = [file.strip('.txt') for file in txts]


'''
This file takes all .txt and .ann and:
.txt: read lines, pull sections Chief Complaint and Present Illness History
    compile into df, drop where both are empty
.ann: read lines, for each patient split into two dfs, one with
    entities, and one with relations between entities
    (get_entities_relations) - can import and use to get dfs for an individual
    then convert into a set of lines containing all Reasons, Drugs and Reason-Drug
    relations (1 line per patient) into ann_train.txt

'''

def load_txt(txt=None, i=None):
    '''
    Pass text file name or index to load file
    '''
    if txt is not None:
        with open(folder + txt) as f:
            lines = f.readlines()
        return lines
    elif i is not None:
        with open(folder + txts[i]) as f:
            lines = f.readlines()
        return lines
    
def load_txt(txt=None, i=None):
    '''
    Pass text file name or index to load file
    '''
    if txt is not None:
        with open(folder + txt) as f:
            lines = f.readlines()
        return lines
    elif i is not None:
        with open(folder + txts[i]) as f:
            lines = f.readlines()
        return lines



def get_entities_relations(patient_id=None, i=None, lower=True):
    '''
    Read .ann file and convert to dataframe of entities and relations
    '''
    if patient_id is not None:
        ann = patient_id + '.ann'
    elif i is not None:
        ann = anns[i]
    lines = load_txt(ann)

    entities = []
    relations = []

    for line in lines:
        cols = line.strip('\n').strip('\t').split('\t')
        if len(cols) == 2:
            try:
                terms = cols[1].split(' ')
                label = terms[0]
                rel1 = terms[1]
                rel2 = terms[2]
                relations.append({
                    'id': cols[0],
                    'label': label,
                    'rel1': rel1,
                    'rel2': rel2
                })
            except:
                #skip rows that have unknown formatting
                pass

        else:
            try:
                terms = cols[1].split(' ')
                label = terms[0]
                start = terms[1]
                stop = terms[2]
                entities.append({
                    'id': cols[0],
                    'label': label,
                    'start': start,
                    'stop': stop,
                    'value': cols[2]
                })
            except:
                #skip rows that have unknown formatting
                pass


        idx = line.split('\t')[0]
        line = line.split('\t')

    entities = pd.DataFrame(entities).set_index('id')
    relations = pd.DataFrame(relations).set_index('id')

    if lower:
        entities['value'] = entities['value'].str.lower()
    return entities, relations




def get_reason_drug(entities, relations):
    '''Pull Reason, Drug from entities, Reason-Drug from relations'''
    reasons = list(entities[entities['label'] == 'Reason']['value'])
    drugs = list(entities[entities['label'] == 'Drug']['value'])
    reason_drug = relations[relations['label'] == 'Reason-Drug']
    
    return reasons, drugs, reason_drug
    
def get_reason_drug_pairs(entities, reason_drug):
    '''Get references from Reason-Drug relations and turn into
    tuples of reason and drug'''
    reason_drug_pairs = []
    for row in reason_drug.iterrows():
        id1 = row[1]['rel1'].split('Arg1:')[1]
        id2 = row[1]['rel2'].split('Arg2:')[1]


        reason = entities.loc[id1, 'value']
        drug = entities.loc[id2, 'value']

        reason_drug_pairs.append((reason, drug))
    return reason_drug_pairs


def convert_to_lines(reasons, drugs, reason_drug_pairs):
    '''
    Convert reasons and drugs into lines
    '''
    reasons = [(' <reas> ' + reason)
                   for reason in reasons]
    drugs = [(' <drug> ' + drug) 
                 for drug in drugs]
    pairs = [(' <pair> ' + reason + ' <sep> ' + drug) 
                 for reason, drug in reason_drug_pairs]
    # reasons = '<reas>' + ' '.join(np.unique(reasons))
    # drugs = '<drug>' + ' '.join(np.unique(drugs))
    # pairs = '<pair>' + '. '.join([reason + ', ' + drug for reason, drug in 
    #                        reason_drug_pairs])
    return reasons, drugs, pairs


    
    


def main():
    

    #Get chief complaint and history from medical record
    #Save to patient_record_summary.csv
    patient_text_rows = []
                
    for patient_id in ids:
        patient_text = {}
        
        txt = patient_id + '.txt'
        lines = load_txt(txt)
        
        # get chief complaint
        chief_complaint = ''
        chief_complaint_idx = -1
        chief_complaint_end_idx = -1
        
        # start and end of chief complaint
        for i, line in enumerate(lines):
            if 'chief complaint' in line.lower():
                chief_complaint_idx = i
                break
        for j, line in enumerate(lines[i:]):
            if line == '\n':
                chief_complaint_end_idx = j + i
                break
        
        # trim \n and join
        if chief_complaint_idx > -1:
            chief_complaint = lines[chief_complaint_idx:chief_complaint_end_idx]
            chief_complaint = [line.strip('\n') for line in chief_complaint]
            chief_complaint = ' '.join(chief_complaint)
            
        patient_text['chief_complaint'] = chief_complaint

        
        # get present history
        present_history = ''
        present_history_idx = -1
        present_history_end_idx = -1
        
        for i, line in enumerate(lines):
            if 'history of present illness' in line.lower():
                present_history_idx = i
                break
        for j, line in enumerate(lines[i+3:]):
            #find the next header
            if ':' in line:
                present_history_end_idx = j + i + 3
                break
        
        if present_history_idx > -1:
            present_history = lines[present_history_idx:present_history_end_idx]
            present_history = [line.strip('\n') for line in present_history]
            present_history = ' '.join(present_history)
        
        patient_text['present_history'] = present_history
        
        
        patient_text_rows.append(patient_text)
        
    patient_df = pd.DataFrame(patient_text_rows, index=ids)
    patient_df = patient_df.drop(
        patient_df[((patient_df['chief_complaint'] == '') & 
         (patient_df['present_history'] == ''))].index
    )
    patient_df = patient_df.fillna('')

    patient_df.to_csv('patient_record_summary.csv')

    print('Saved summary of Chief Complaints and History of Present Illness to patient_record_summary.csv')


    # Go through all the .ann files to collect drug, reason and drug_reason pairs
    # Build into a training database into memory
    # Build into a list of all_reason_drug_pairs lines to train tokenizer
    all_reason_drug_pairs = []
    training_lines = []
    for i in tqdm(range(len(anns))):
        entities, relations = get_entities_relations(i=i)
        reasons, drugs, reason_drug = get_reason_drug(entities, relations)
        reason_drug_pairs = get_reason_drug_pairs(entities, reason_drug)
        reasons, drugs, reason_drug_pairs = \
            convert_to_lines(reasons, drugs, reason_drug_pairs)
        
        # used to build vocab
        all_reason_drug_pairs += reason_drug_pairs
        
        # used to train on
        line = ''.join(reasons) + ' ' + ''.join(drugs) + ' ' + ''.join(reason_drug_pairs) + '</s>'
        training_lines.append(line)

        
    # Convert lines with reasons, drugs, and reason-drug pairs to training_lines
    file =  open('ann_train.txt', 'w')
    for line in training_lines:
        file.write(line + '\n')
    file.close()

    print('Saved reason-drug pair training lines for transformers to ann_train.txt')

if __name__ == '__main__':
    main()