from sklearn import feature_extraction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

patient_df = pd.read_csv('patient_record_summary.csv')
patient_df = patient_df.fillna('')
lines = list(patient_df['chief_complaint'].str.lower() + ' \n ' + patient_df['present_history'].str.lower())

vectorizer = feature_extraction.text.TfidfVectorizer()
vectors = vectorizer.fit_transform(lines)

# Attempt to find ideal number of clusters
# sum_of_squared_dists = []
# K = range(1, 150)
# for k in tqdm(K):
#     km = KMeans(n_clusters=k)
#     km = km.fit(vectors)
#     sum_of_squared_dists.append(km.inertia_)
# K = modelkmeans

# plt.plot(K[:len(sum_of_squared_dists)], sum_of_squared_dists)
# plt.scatter(K[13], sum_of_squared_dists[13])

kmeans = KMeans(3)
cluster_x = kmeans.fit_predict(vectors)
# for i in range(3):
#     print(np.sum(cluster_x == i))

pca = PCA(6)
pca_x = pca.fit_transform(vectors.todense())

plt.figure(figsize=(8,8))
plt.scatter(pca_x.T[0], pca_x.T[1], c=cluster_x)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.savefig('clusters.png', bbox_inches='tight')
plt.show()


id_to_vocab = {}
for key in vectorizer.vocabulary_:
    id_to_vocab[vectorizer.vocabulary_[key]] = key

def print_pca_words(components, cutoff=0.09):
    '''
    Print out words that are most relevant to an axis
    of variation from PCA
    Pass in the components vector from PCA and cutoff of importance
    '''
    idxs = np.where(np.abs(components) > cutoff)[0]
    print([id_to_vocab[i] for i in idxs])
    

plt.figure(figsize=(8,8))
vocab_len = len(pca.components_[1])
plt.scatter(np.arange(vocab_len), np.abs(pca.components_[1]))
plt.plot([0, vocab_len], [0.09, 0.09], '--')
plt.xlabel('Word Index')
plt.ylabel('Importance to PCA')
plt.title('2nd PCA Component Vocab Weights')
plt.savefig('pca_importance.png', bbox_inches='tight')
plt.show()

for i in range(5):
    print('Words important for the {}th pca component'.format(i+1))
    print_pca_words(pca.components_[i])


chest_idx = vectorizer.vocabulary_['chest']
abdominal_idx = vectorizer.vocabulary_['abdominal']
print('Weight of word "chest" in PCA3')
print(pca.components_[2][chest_idx])
print('Weight of word "abdominal" in PCA3')
print(pca.components_[2][abdominal_idx])