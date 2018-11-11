import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
class DBSCANClusterer():
    def __init__(self,fileName,minPts=10,eps=0.3,target_header = "Target"):
        self.fileName = fileName
        self.minPts = minPts
        self.eps = eps
        self.dbscanObj = DBSCAN(self.eps, self.minPts)
        self.target_header = target_header

    def getOutputHeaders(self):
        data = pd.read_csv(self.fileName).columns
        data =data.tolist()
        data.append(self.target_header)
        return data

    def cluster(self):
        X=self.load_data(self.fileName)
        print('Clustering Initiated', np.shape(X))
        result = self.dbscanObj.fit(X)
        labels = result.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
        core_samples_mask[result.core_sample_indices_] = True
        unique_labels = set(labels)

        Y = None
        for k in unique_labels:
            class_member_mask = (labels == k)
            M = X[class_member_mask & core_samples_mask]
            for data in M:
                new_data = np.append(data, np.array(k))
                if not hasattr(Y, 'shape'):
                    Y = new_data
                else:
                    Y = np.vstack((Y, new_data))
        print('Clustering Complete', n_clusters_)
        return Y,n_clusters_

    def load_data(self,fileName):
        M = np.genfromtxt(fileName,delimiter=',',skip_header=True)
        return M

    def cluster_and_save(self,outputFile):
        Y,n_clusters_ = self.cluster()
        np.savetxt(outputFile,Y,delimiter=',',fmt='%.3e',header=','.join(self.getOutputHeaders()),comments='')


