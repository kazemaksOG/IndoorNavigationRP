from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from threading import Lock
from utils import create_confusion_matrix
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import tensorflow as tf

knn_param_grid = {
    'n_neighbors': [1,2,3,4,5,6,7,8,9,10],  
    'weights': ['uniform', 'distance']
}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovo', 'ovr']
}


class WifiClassifier:
    def __init__(self, consistency_type=None):
        self.int_to_label = []
        self.model = None
        self.training_lock = Lock()
        self.model_trained = False
        self.consistency_list = []
        self.consistency_type = None
        self.cross_entropy = 0
        self.EWM = 0
    def train(self, dataset, int_to_label, room_amount, filename=None):
        train_set, train_labels, validation_set, validation_labels= dataset
        self.int_to_label = int_to_label
        with self.training_lock:
            if filename == None:
                knn_model = KNeighborsClassifier()
                knn_grid_search = GridSearchCV(knn_model, knn_param_grid)
                knn_grid_search.fit(train_set, train_labels)

                svm_model = svm.SVC(probability=True)
                svm_grid_search = GridSearchCV(svm_model, svm_param_grid)
                svm_grid_search.fit(train_set, train_labels)
                
                best_knn_params = knn_grid_search.best_params_
                best_svm_params = svm_grid_search.best_params_
                best_knn_score = knn_grid_search.best_score_
                best_svm_score = svm_grid_search.best_score_
                print("KNN PARAMS: " + str(best_knn_params) + "\nACCURACY: " + str(best_knn_score))
                print("SVM PARAMS: " + str(best_svm_params) + "\nACCURACY: " + str(best_svm_score))

                if best_knn_score > best_svm_score:
                    print("KNN CHOSEN")
                    self.model = knn_grid_search.best_estimator_
                else:
                    print("SVM CHOSEN")
                    self.model = svm_grid_search.best_estimator_

                # knn_results = knn_grid_search.cv_results_
                # svm_results = svm_grid_search.cv_results_
                # knn_combinations = zip(knn_results['params'], knn_results['mean_test_score'])
                # svm_combinations = zip(svm_results['params'], svm_results['mean_test_score'])
                # for params, accuracy in knn_combinations:
                #     print("Parameters:", params, "Accuracy:", accuracy)
                # for params, accuracy in svm_combinations:
                #     print("Parameters:", params, "Accuracy:", accuracy)


                pickle.dump(self.model, open("./models/best_wifi_4.sav", 'wb'))
            else:
                self.model = pickle.load(open("./models/" + filename, 'rb'))

                 
            self.model_trained = True
            self.create_consistency_list(train_set, tf.keras.utils.to_categorical(train_labels, room_amount) )

            # wifi_predictions = []
            # for sample in validation_set:
            #     wifi_predictions.append(self.model.predict(np.reshape(sample,(1, len(sample))))[0])
            # print(wifi_predictions)
            # cm = confusion_matrix(validation_labels, wifi_predictions)
            # consistency_list = []
            # for true_label, predictions in enumerate(cm):
            #     true_positives = 0
            #     false_positives = 0
            #     for predicted_label, amount in enumerate(predictions):
            #         if predicted_label == true_label:
            #             true_positives += amount
            #         else:
            #             false_positives += amount
            #     consistency_list.append(round(true_positives / (false_positives + true_positives),1) )
            # print(consistency_list)
            # self.consistency_list = consistency_list
            # r_list = []
            # for acc in consistency_list:
            #     r_list.append(np.count_nonzero(consistency_list == acc))
            # print(r_list)
            # E = 0

            # above_80_total_sum = 0
            # for i, acc in enumerate(consistency_list):
            #     if acc >= 0.8:
            #         above_80_total_sum += np.count_nonzero(consistency_list == acc)

            # print("above 80:" + str(above_80_total_sum))

            # for i, acc in enumerate(consistency_list):
            #     pi = r_list[i] / np.sum(r_list)
            #     wi = 0
            #     if acc >= 0.8:
                    
            #         wi = 1 - (np.count_nonzero(consistency_list == acc) / above_80_total_sum)
            #         print(wi)
            #     else:
            #         wi = 1
            #     E -= wi * pi * np.log2(pi)
            # print("E:" + str(E))

            # self.EWM = E
    def create_consistency_list(self, validation_set, validation_labels):

        wifi_predictions = []
        for sample in validation_set:
            wifi_predictions.append(self.classify_probability(np.reshape(sample, (1,len(sample))))[0])
      
        cce = tf.keras.losses.CategoricalCrossentropy()
        self.cross_entropy = cce(validation_labels, wifi_predictions).numpy()
        print(cce(validation_labels, wifi_predictions).numpy())



        # cm = confusion_matrix(validation_labels, validation_predictions)
        # entropy_list = []
        # for true_label, predictions in enumerate(cm):
        #     true_positives = 0
        #     false_positives = 0
        #     summ = np.sum(predictions)
        #     entropy_list.append(max(0, 1 - entropy(predictions / summ)))
        #     for predicted_label, amount in enumerate(predictions):
        #         if predicted_label == true_label:
        #             true_positives += amount
        #         else:
        #             false_positives += amount
        #     consistency_list.append( true_positives / (false_positives + true_positives))
        
        # entrop = entropy(consistency_list)
        # print(entrop)


    def classify(self, sample):
        sample = np.asarray(sample)
        if self.model_trained:
            return self.int_to_label[self.model.predict(np.reshape(sample,(1, len(sample))))[0]]
        else:
            return "Model is not trained yet"

    def test_accuracy(self, tests, labels):
        if self.model_trained:
            wifi_predictions = []
            for sample in tests:
                wifi_predictions.append(self.model.predict(np.reshape(sample,(1, len(sample))))[0])
            accuracy = accuracy_score(labels, wifi_predictions)
            create_confusion_matrix(labels, wifi_predictions, np.asarray(self.int_to_label),accuracy, "Wifi localization")
            return accuracy
        else:
            return "Model is not trained yet"

    def classify_probability(self, sample):

        if self.model_trained:
            probabilities = self.model.predict_proba(sample)
            if self.consistency_type != None:
                return probabilities * self.consistency_list
            else:
                return probabilities

        else:
            return "Model is not trained yet"

    def get_int_to_label(self):
        return np.asarray(self.int_to_label)




if __name__ == "__main__":
    
    wifis, wifi_labels, wifi_int_to_label = db.get_wifi_training_set()
    room_amount = db.get_room_amount()
    wifi_dataset = train_test_split(wifis, wifi_labels, test_size=test_split, random_state=42)
    acoustic_model.train(acoustic_dataset, image_int_to_label, room_amount)
    wifi_model.train(wifi_dataset, wifi_int_to_label, room_amount)

    test_classifiers(acoustic_dataset[1], acoustic_dataset[3], wifi_dataset[1], wifi_dataset[3])