# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle,os


class TrainClassifier():
    def __init__(self,data,headers,target_header='Target',train_test_split=0.7):
        self.train_test_split = train_test_split
        self.headers = headers
        self.input_data=data
        self.target_header = target_header
        self.model = None

    def read_data(self,path):
        data = pd.read_csv(path)
        return data

    def split_dataset(self,dataset, train_percentage, feature_headers, target_header):
        # Split dataset into train and test dataset

        train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                            train_size=train_percentage,test_size=1-train_percentage)
        return train_x, test_x, train_y, test_y

    def handle_missing_values(self,dataset, missing_values_header, missing_label):
        return dataset[dataset[missing_values_header] != missing_label]

    def random_forest_classifier(self,features, target,n_estimators=101,max_features_for_split='auto'):
        clf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features_for_split)
        clf.fit(features, target)
        return clf

    def get_cluster_file_path(self):
        return self.input_data[:self.input_data.rfind(os.sep)]

    def dataset_statistics(self,dataset):
        print dataset.describe()


    def train(self):
        dataset = pd.read_csv(self.input_data)
        #self.dataset_statistics(dataset)
        missing_value = -1 # for noise data
        dataset = self.handle_missing_values(dataset, self.target_header, missing_value)
        train_x, test_x, train_y, test_y = self.split_dataset(dataset, 0.7, self.headers[:-1], self.target_header)
        print('Training begun on :',train_x.shape)
        trained_model = self.random_forest_classifier(train_x, train_y)
        print('Training completed!')
        predictions = trained_model.predict(test_x)
        test_acc = accuracy_score(test_y, predictions)
        print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
        self.model = trained_model
        return trained_model,test_acc

    def train_and_save(self):
        trained_model,acc = self.train()
        file_path = os.path.join(self.get_cluster_file_path(),(self.target_header+'_'+str(round(acc,3))+".model"))
        print(file_path)
        with open(file_path, 'wb') as file:
            pickle.dump(trained_model,file)

    def predict(self,data):
        if self.model is None:
            raise Exception('Model not present!')
        return self.model.predict(data)

    def load_model(self,file_path):
        with open(file_path, 'rb') as file:
            self.model= pickle.load(file)
        print('Model Loaded!')