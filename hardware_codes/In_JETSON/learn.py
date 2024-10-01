'''
Universität Siegen
Naturwissenschaftlich-Technische Fakultät
Department Elektrotechnik und Informatik
Lehrstuhl für Regelungs- und Steuerungstechnik

Master Thesis: Vision-aided Automated Driving using Panoramic View
Done By: Sourav Poudyal, 1607167
First Examiner: Prof. Dr.-Ing. habil. Michael Gerke
Supervisior: Dr.-Ing. Nasser Gyagenda
'''

#This is custome implementation of DBSCAN 

#####################
import numpy as np

class customMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
    
    def fit(self, X):
        X = np.array(X)
        self.min_ = X.min(axis=0)
        self.scale_ = (X.max(axis=0) - X.min(axis=0))
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)  # Avoid division by zero
        return self
    
    def transform(self, X):
        X = np.array(X)
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class customDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1
        
        self.labels_ = labels
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def _region_query(self, X, idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors
    
    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id):
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1



