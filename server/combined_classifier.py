import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from utils import create_confusion_matrix
from threading import Lock
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



class DynamicEntropyModel:
    def __init__(self, acoustic_model, wifi_model, int_to_label, room_amount):
        self.int_to_label = int_to_label
        self.acoustic_model = acoustic_model
        self.wifi_model = wifi_model
        self.training_lock = Lock()
        self.model_trained = False
        self.room_amount = room_amount

    def set_int_to_label(self,int_to_label):
        self.int_to_label = int_to_label

    def train(self, acoustic_training_dataset, wifi_training_dataset, acoustic_test_dataset, wifi_test_dataset):
        with self.training_lock:
            acoustic_test_set, acoustic_test_label = acoustic_test_dataset
            wifi_test_set, wifi_test_label = wifi_test_dataset
            
            test_accuracy = self.test_accuracy(acoustic_test_set, wifi_test_set, wifi_test_label, True)
            return test_accuracy

    def calculate_entropy_and_prediction(self, acoustic_sample_batch, wifi_sample_batch):
        acoustic_predictions = []
        wifi_predictions = []

        for acoustic_sample, wifi_sample in zip(acoustic_sample_batch, wifi_sample_batch):
            acoustic_prediction = np.argmax(self.acoustic_model.get_predictions(acoustic_sample)[0])
            wifi_prediction = self.wifi_model.model.predict(np.reshape(wifi_sample,(1, len(wifi_sample))))[0]
            acoustic_predictions.append(acoustic_prediction)
            wifi_predictions.append(wifi_prediction)


        wifi_prediction = np.argmax(np.bincount(wifi_predictions))
        acoustic_prediction = np.argmax(np.bincount(acoustic_predictions))
        
        acoustic_probability = (np.bincount(acoustic_predictions) / len(acoustic_predictions))
        wifi_probability = (np.bincount(wifi_predictions) / len(wifi_predictions))
        # print(wifi_probability)
        # print(acoustic_probability)
        acoustic_entropy = entropy(acoustic_probability)
        wifi_entropy = entropy(wifi_probability)
        combined_prediction_less_entropy = 0
        if acoustic_entropy > wifi_entropy:
            combined_prediction_less_entropy = wifi_prediction
        else:
            combined_prediction_less_entropy = acoustic_prediction

        acoustic_prediction_weight = wifi_entropy / (acoustic_entropy + wifi_entropy)
        wifi_prediction_weight = acoustic_entropy / (acoustic_entropy + wifi_entropy)

        # combined_prediction_probability = acoustic_prediction_weight * acoustic_prediction_probability + wifi_prediction_weight * wifi_prediction_probability
        # combined_prediction = np.argmax(combined_prediction_probability)

        return combined_prediction_less_entropy, acoustic_prediction, wifi_prediction

    def classify(self, acoustic_sample, wifi_sample):
        return self.int_to_label[self.get_prediction(acoustic_sample, wifi_sample)]

    def test_accuracy(self, acoustic_test, wifi_test, labels, save_matrix):
        print(labels)
        batch_list = []
        index = 0
        np_bin = np.bincount(labels)
        batch_labels = []
        wifi_predictions = []
        acoustic_predictions = []
        combined_predictions = []
        for (label, binned) in enumerate(np_bin):
            batch_labels.append(label)
            wifi_batch = []
            acoustic_batch = []
            for label in range(0,binned):
                wifi_batch.append(wifi_test[index])
                acoustic_batch.append(acoustic_test[index])
                index += 1

            combined_prediction, acoustic_prediction, wifi_prediction = self.calculate_entropy_and_prediction(acoustic_batch, wifi_batch)
            combined_predictions.append(combined_prediction)
            acoustic_predictions.append(acoustic_prediction)
            wifi_predictions.append(wifi_prediction)

        accuracy = accuracy_score(batch_labels, combined_predictions)
        wifi_acc = accuracy_score(batch_labels, wifi_predictions)
        acoustic_acc = accuracy_score(batch_labels, acoustic_predictions)
        print("batched wifi accuracy: " + str(wifi_acc))
        print("batched acoustic accuracy: " + str(acoustic_acc))
        if save_matrix:
            create_confusion_matrix(batch_labels, combined_predictions, self.int_to_label, accuracy, "Dynamic entropy")

        return accuracy





logistic_param_grid = {
    'penalty': ['none', 'l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

knn_param_grid = {
    'n_neighbors': [6,7,8,9,10],  
    'weights': ['distance']
}

svm_param_grid = {
    'C': [10],
    'kernel': ['linear'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovo', 'ovr']
}


class StackingModel:
    def __init__(self, acoustic_model, wifi_model, int_to_label):
        self.int_to_label = int_to_label
        self.acoustic_model = acoustic_model
        self.wifi_model = wifi_model
        self.training_lock = Lock()
        self.model_trained = False
        self.prediction_model = None

    
    def set_int_to_label(self,int_to_label):
        self.int_to_label = int_to_label

    def train(self, acoustic_training_dataset, wifi_training_dataset, acoustic_test_dataset, wifi_test_dataset):
        with self.training_lock:
            acoustic_train_set, acoustic_train_label = acoustic_training_dataset
            acoustic_test_set, acoustic_test_label = acoustic_test_dataset
            wifi_train_set, wifi_train_label = wifi_training_dataset
            wifi_test_set, wifi_test_label = wifi_test_dataset

            sc = StandardScaler()
            combined_train_set = self.get_combined_set(acoustic_train_set, wifi_train_set)
            combined_train_set = sc.fit_transform(combined_train_set)
            combined_test_set = self.get_combined_set(acoustic_test_set, wifi_test_set)
            combined_test_set = sc.transform(combined_test_set)

            knn_model = KNeighborsClassifier()
            knn_grid_search = GridSearchCV(knn_model, knn_param_grid)
            knn_grid_search.fit(combined_train_set, acoustic_train_label)

            svm_model = svm.SVC(probability=True)
            svm_grid_search = GridSearchCV(svm_model, svm_param_grid)
            svm_grid_search.fit(combined_train_set, acoustic_train_label)
            
            best_knn_params = knn_grid_search.best_params_
            best_svm_params = svm_grid_search.best_params_
            best_knn_score = knn_grid_search.best_score_
            best_svm_score = svm_grid_search.best_score_
            print("KNN PARAMS: " + str(best_knn_params) + "\nACCURACY: " + str(best_knn_score))
            print("SVM PARAMS: " + str(best_svm_params) + "\nACCURACY: " + str(best_svm_score))

            logistic_model = LogisticRegression(random_state=42, max_iter=1000, penalty=None)
            logistic_grid_search = GridSearchCV(logistic_model, logistic_param_grid)
            logistic_grid_search.fit(combined_train_set, acoustic_train_label)

            best_logistic_params = logistic_grid_search.best_params_
            best_logistic_score = logistic_grid_search.best_score_
            print("logistic params: " + str(best_logistic_params) + " accuracy: " + str(best_logistic_score))
            self.prediction_model = logistic_grid_search.best_estimator_
            predictions = self.prediction_model.predict(combined_test_set)
            print(predictions)
            print(wifi_test_label)
            accuracy = accuracy_score(predictions, wifi_test_label)
            create_confusion_matrix(wifi_test_label, predictions, self.int_to_label, accuracy, "Stacking" )
            return accuracy

    def get_combined_set(self, acoustic_train, wifi_train):
        combined_predictions = []
        for acoustic_sample, wifi_sample in zip(acoustic_train, wifi_train):
            acoustic_prediction = np.argmax(self.acoustic_model.get_predictions(acoustic_sample)[0])
            wifi_prediction = self.wifi_model.model.predict(np.reshape(wifi_sample,(1, len(wifi_sample))))[0]
            combined_predictions.append(np.array((acoustic_prediction, wifi_prediction)))
        
        combined_predictions = np.asarray(combined_predictions)
        return combined_predictions


class EntropyWeightModel:
    def __init__(self, acoustic_model, wifi_model, acoustic_entropy, wifi_entropy, int_to_label ,entropy_type=""):
        self.int_to_label = int_to_label
        self.acoustic_model = acoustic_model
        self.wifi_model = wifi_model
        self.training_lock = Lock()
        self.model_trained = False
        self.wifi_entropy = wifi_entropy
        self.acoustic_entropy = acoustic_entropy
        self.entropy_type = entropy_type
    
    def set_int_to_label(self,int_to_label):
        self.int_to_label = int_to_label

    def train(self, acoustic_training_dataset, wifi_training_dataset, acoustic_test_dataset, wifi_test_dataset):
        with self.training_lock:
            acoustic_train_set, acoustic_train_label = acoustic_training_dataset
            acoustic_test_set, acoustic_test_label = acoustic_test_dataset
            wifi_train_set, wifi_train_label = wifi_training_dataset
            wifi_test_set, wifi_test_label = wifi_test_dataset
            
            test_accuracy = self.entropy_weight_test_accuracy(acoustic_test_set, wifi_test_set, wifi_test_label, True)
            return test_accuracy

    def get_prediction(self, acoustic_sample, wifi_sample):
        acoustic_prediction = self.acoustic_model.get_predictions(acoustic_sample)[0]
        wifi_prediction = self.wifi_model.classify_probability(np.reshape(wifi_sample, (1,len(wifi_sample))))[0]

        combined_probability = self.acoustic_entropy * acoustic_prediction + wifi_prediction * self.wifi_entropy
        combined_prediction = np.argmax(combined_probability)
        
        return combined_prediction

    def classify(self, acoustic_sample, wifi_sample):
        return self.int_to_label[self.get_prediction(acoustic_sample, wifi_sample)]

    def entropy_weight_test_accuracy(self, acoustic_test, wifi_test, labels, save_matrix):
        combined_predictions = []
        for acoustic_sample, wifi_sample in zip(acoustic_test, wifi_test):
            combined_predictions.append(self.get_prediction(acoustic_sample, wifi_sample))
        combined_predictions = np.asarray(combined_predictions)

        accuracy = accuracy_score(labels, combined_predictions)
        if save_matrix:
            create_confusion_matrix(labels, combined_predictions, self.int_to_label, accuracy, "Weighted cross entropy ")

        return accuracy


class WeightedAverage:
    def __init__(self, acoustic_model, wifi_model, int_to_label):
        self.int_to_label = int_to_label
        self.acoustic_model = acoustic_model
        self.wifi_model = wifi_model
        self.training_lock = Lock()
        self.model_trained = False
        self.weight = -1
    
    def set_int_to_label(self,int_to_label):
        self.int_to_label = int_to_label

    def train(self, acoustic_training_dataset, wifi_training_dataset, acoustic_test_dataset, wifi_test_dataset):
        with self.training_lock:
            acoustic_train_set, acoustic_train_label = acoustic_training_dataset
            acoustic_test_set, acoustic_test_label = acoustic_test_dataset
            wifi_train_set, wifi_train_label = wifi_training_dataset
            wifi_test_set, wifi_test_label = wifi_test_dataset

            best_weight = 0.4
            best_accuracy = "NAN"
            best_weight = -1
            best_accuracy = -1
            for weight in np.arange(0, 5, 0.1):
                accuracy = self.weighted_average_test_accuracy(acoustic_train_set, wifi_train_set, wifi_train_label, weight, False)
                if best_accuracy < accuracy:
                    best_weight = weight
                    best_accuracy = accuracy
            
            self.weight = best_weight
            self.model_trained = True
            test_accuracy = self.weighted_average_test_accuracy(acoustic_test_set, wifi_test_set, wifi_test_label, best_weight, True)
            print("BEST WEIGHT:" + str(best_weight) + " TRAIN_ACCURACY:" + str(best_accuracy) + " TEST_ACCURACY:" + str(test_accuracy))
            return test_accuracy

    def get_prediction(self, acoustic_sample, wifi_sample, acoustic_weight=None):
        if acoustic_weight == None:
            acoustic_weight = self.weight
        acoustic_prediction = self.acoustic_model.get_predictions(acoustic_sample)[0]
        wifi_prediction = self.wifi_model.classify_probability(np.reshape(wifi_sample, (1,len(wifi_sample))))[0]

        combined_probability = acoustic_weight * acoustic_prediction + wifi_prediction
        combined_prediction = np.argmax(combined_probability)
        
        return combined_prediction

    def classify(self, acoustic_sample, wifi_sample):
        return self.int_to_label[self.get_prediction(acoustic_sample, wifi_sample)]

    def weighted_average_test_accuracy(self, acoustic_test, wifi_test, labels, acoustic_weight, save_matrix):
        combined_predictions = []
        for acoustic_sample, wifi_sample in zip(acoustic_test, wifi_test):
            combined_predictions.append(self.get_prediction(acoustic_sample, wifi_sample, acoustic_weight))
        combined_predictions = np.asarray(combined_predictions)

        accuracy = accuracy_score(labels, combined_predictions)
        if save_matrix:
            create_confusion_matrix(labels, combined_predictions, self.int_to_label, accuracy, "Weighted average")

        return accuracy


class TwoStep:
    def __init__(self, acoustic_model, wifi_model, int_to_label):
        self.int_to_label = int_to_label
        self.acoustic_model = acoustic_model
        self.wifi_model = wifi_model
        self.training_lock = Lock()
        self.model_trained = False
        self.top_k = -1
    
    def set_int_to_label(self, int_to_label):
        self.int_to_label = int_to_label
    def train(self, acoustic_training_dataset, wifi_training_dataset, acoustic_test_dataset, wifi_test_dataset):
        with self.training_lock:
            acoustic_train_set, acoustic_train_label = acoustic_training_dataset
            acoustic_test_set, acoustic_test_label = acoustic_test_dataset
            wifi_train_set, wifi_train_label = wifi_training_dataset
            wifi_test_set, wifi_test_label = wifi_test_dataset
            
            best_k = 2
            best_accuracy = "NAN"
            best_k = 2
            best_accuracy = -1
            for k in range(2, len(self.int_to_label) + 1):
                accuracy = self.two_step_test_accuracy(acoustic_train_set, wifi_train_set, wifi_train_label, k, False)
                if best_accuracy < accuracy:
                    best_k = k
                    best_accuracy = accuracy
            
            test_accuracy = self.two_step_test_accuracy(acoustic_test_set, wifi_test_set, wifi_test_label, best_k, True)
            print("BEST K:" + str(best_k) + " TRAIN_ACCURACY:" + str(best_accuracy) + " TEST_ACCURACY:" + str(test_accuracy))
            self.top_k = best_k
            self.model_trained = True
            return test_accuracy

    def get_prediction(self, acoustic_sample, wifi_sample, top_k=None):
        if top_k == None:
            top_k = self.top_k
        
        wifi_top = wifi_top_k(self.wifi_model, top_k, wifi_sample)
        acoustic_top = acoustic_top_k(self.acoustic_model, 1000, acoustic_sample)
        for top_choice in acoustic_top:
            if np.any(wifi_top == top_choice):
                return top_choice

    def classify(self, acoustic_sample, wifi_sample):
        return self.int_to_label[self.get_prediction(acoustic_sample, wifi_sample)]

    def two_step_test_accuracy(self, acoustic_test, wifi_test, labels, top_k, save_matrix):

        combined_predictions = []
        for acoustic_sample, wifi_sample in zip(acoustic_test, wifi_test):
            combined_predictions.append(self.get_prediction(acoustic_sample, wifi_sample, top_k))
        combined_predictions = np.asarray(combined_predictions)

        accuracy = accuracy_score(labels, combined_predictions)
        if save_matrix:
            create_confusion_matrix(labels, combined_predictions, self.int_to_label, accuracy, "2-step localization")



        return accuracy


def wifi_top_k(wifi_model, top_k, wifi_sample):
    wifi_prediction = wifi_model.classify_probability(np.reshape(wifi_sample, (1,len(wifi_sample))))[0]
    wifi_top = wifi_prediction.argsort()[-top_k:][::-1]
    return wifi_top

def acoustic_top_k(acoustic_model, top_k, acoustic_sample):
    acoustic_prediction = acoustic_model.get_predictions(acoustic_sample)[0]
    acoustic_top = acoustic_prediction.argsort()[-top_k:][::-1]
    return acoustic_top

def wifi_top_k_test_accuracy(wifi_model, top_k, wifi_test, labels):
    int_to_label = wifi_model.get_int_to_label()
    wifi_top = []
    for wifi_sample in wifi_test:
        wifi_top.append(wifi_top_k(wifi_model, top_k, wifi_sample))
    combined_predictions = []
    for i, top_list in enumerate(wifi_top):
        if np.any(top_list == labels[i]):
            combined_predictions.append(labels[i])
        else:
            combined_predictions.append(top_list[0])
    accuracy = accuracy_score(labels, combined_predictions)

    create_confusion_matrix(labels, combined_predictions, int_to_label, accuracy, "Top "+ str(top_k) +" wifi")
    
    return accuracy


def acoustic_top_k_test_accuracy(acoustic_model, top_k, acoustic_test, labels):
    int_to_label = acoustic_model.get_int_to_label()
    acoustic_top = []
    for acoustic_sample in acoustic_test:
        acoustic_top.append(acoustic_top_k(acoustic_model, top_k, acoustic_sample))
    combined_predictions = []
    for i, top_list in enumerate(acoustic_top):
        if np.any(top_list == labels[i]):
            combined_predictions.append(labels[i])
        else:
            combined_predictions.append(top_list[0])
    accuracy = accuracy_score(labels, combined_predictions)

    create_confusion_matrix(labels, combined_predictions, int_to_label, accuracy, "Top "+ str(top_k) +" acoustic")
    
    return accuracy



def wifi_top_k_to_string(wifi_model, top_k, wifi_sample):
    int_to_label = wifi_model.get_int_to_label()
    wifi_prediction = wifi_model.classify_probability(np.reshape(wifi_sample, (1,len(wifi_sample))))[0]
    wifi_top = wifi_prediction.argsort()[-top_k:][::-1]
    accuracy = np.sort(wifi_prediction)[::-1]
    string_list = []
    for wifi_label, acc in zip(wifi_top, accuracy):
        string_list.append(int_to_label[wifi_label] + " " + str(round(acc, 2)))

    return string_list


def acoustic_top_k_to_string(acoustic_model, top_k, acoustic_sample):
    int_to_label = acoustic_model.get_int_to_label()
    acoustic_prediction = acoustic_model.get_predictions(acoustic_sample)[0]
    acoustic_top = acoustic_prediction.argsort()[-top_k:][::-1]
    accuracy = np.sort(acoustic_prediction)[::-1]
    string_list = []

    for acoustic_label, acc in zip(acoustic_top, accuracy):
        string_list.append(int_to_label[acoustic_label] + " " + str(round(acc, 2)))

    return string_list
