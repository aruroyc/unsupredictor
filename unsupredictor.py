from DBSCANClusterer import *
from Classifier import *
import pandas as pd
import sys
class UnsupervisedLearnPredictor():
    def __init__(self,inputfile,target_header='Target',train_test_split=0.7,minPts=10,eps=0.3):
        self.clusterer = DBSCANClusterer(inputfile,minPts,eps,target_header)
        self.input_file = inputfile
        self.cluster_file = self.get_cluster_file_name()
        self.classifier = TrainClassifier(self.get_cluster_file_name(),self.getInputHeaders(),target_header,train_test_split)


    def getInputHeaders(self):
        data = pd.read_csv(self.cluster_file).columns
        return data

    def get_cluster_file_name(self):
        return self.input_file[:-4]+'_cluster.csv'

    def _compute_classes_and_save(self):
        self.clusterer.cluster_and_save(self.cluster_file)


    def _train_classifier_and_save(self):
        return self.classifier.train_and_save()

    def learn_classes_and_save_model(self):
        self._compute_classes_and_save()
        self._train_classifier_and_save()

    def predict(self,data):
        return self.classifier.predict(data)

    def load_precomputed_classifier(self,file_path):
        self.classifier.load_model(file_path)