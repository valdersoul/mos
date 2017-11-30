from sklearn.cluster import KMeans
import torch
import numpy as np

class Clustering:
    def __init__(self, em_vectors, clusters):
        super(Clustering, self).__init__()
        if isinstance(em_vectors, torch.nn.parameter.Parameter):
            self.em_vectors = em_vectors.data.cpu().numpy()
        else: #only for Parameter or np.array
            self.em_vectors = em_vectors
        self.clusters = clusters
    def fit(self):
        kmeans = KMeans(n_clusters=self.clusters, random_state=0).fit(self.em_vectors)
        return kmeans

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    words = []
    embeddings = []
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        words.append(word)
        embeddings.append(embedding)
    print ("Done.",len(words)," words loaded!")
    return words, embeddings

if __name__ == '__main__':
    glove_file = '/home/zheng/pcProjects/allennlp/data/glove.6B.100d.txt'
    words, embeddings = loadGloveModel(gloveFile=glove_file)
    em_matrix = np.array(embeddings)
    cluster = Clustering(em_vectors=em_matrix, clusters=100)
    clustered_embedding = cluster.fit()
    labels = clustered_embedding.labels_
