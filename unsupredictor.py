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
        data = pd.read_csv(self.input_file).columns
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
if __name__ == '__main__':
    trainer = UnsupervisedLearnPredictor(sys.argv[1],minPts=500,eps=50)
    trainer.learn_classes_and_save_model()