from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(csv):
    model = textual_model()
    df_clean = model.load_csv(csv)
    df2 = model.load_medical_history(df_clean)
    df3 = model.load_patient_symptoms(df_clean)
    df4 = model.merge_df2_df3(df2, df3)
    upd_df4 = model.gen_smok_encoding(df4)
    df5, target_labels = model.merge_dataframes(upd_df4)

    return df5, target_labels



def predict(csv, model_sav):
    """
    Sample file for the API of individual models

    Inputs:
    - df: A pandas dataframe of shape (m, ?) containing the input data
    - model_sav: The best trained model file for this class

    Outputs:
    - A numpy array of shape (m, 1) outputting the predicted probability of
      COVID-19 by this model
    """

    model = textual_model()
    result = model.test_data(csv, model_sav)
    # returns only the positive probability of the first sample
#     return 100*result[0][1]
    return result

def train(csv, save_model_name):
    """
    For training the model on new coming data.
    
    Inputs:
    - csv: an updated csv file to train model on 
    - save_model_name: takes in filename in which u want to save your model, note: filename should end up with .sav 
    Outputs:
    - outputs the filename in which your model is stored
    
    """
    model = textual_model()
    trained_model = model.train_callable(csv, save_model_name)
    return trained_model    

class textual_model:
    def __init__(self):
        pass

    def load_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        
        df_clean = df.drop(columns=["seq_id", "patient_id", "date" , "cough_filename", "finger_filename",
                                    "patient_smartphone", "breathing_filename"])

        ff = df_clean["medical_history"].isna().sum()

        # Filling nan values with None.
        df_clean["medical_history"].fillna("None,", inplace = True) 
        df_clean["smoker"].fillna("no", inplace = True) 
        df_clean["patient_reported_symptoms"].fillna("None,", inplace = True) 
        return df_clean
    
    def load_test_csv(self, df):
        # df = pd.read_csv(csv_path)

        # df_clean = df.drop(columns=["seq_id", "patient_id", "date" , "cough_filename", "finger_filename",
        #                                 "patient_smartphone", "breathing_filename"])

        df_clean = df.head(1)
        
        ff = df_clean["medical_history"].isna().sum()

        # Filling nan values with None.
        df_clean["medical_history"].fillna("None,", inplace = True) 
#         df_clean["smoker"].fillna("no", inplace = True) 
        df_clean["patient_reported_symptoms"].fillna("None,", inplace = True) 
        return df_clean

    def load_medical_history(self, df_clean):
        top_medical_history = ['None,', 'Asthma or chronic lung disease,',
       'Disease or conditions that make it harder to cough,',
       'Diabetes with complications,', 'Pregnancy,',
       'Congestive heart failure,', 'Extreme obesity,']

        df2 = df_clean.copy()
        for mh in top_medical_history:
            df2[mh] = df2.medical_history.str.contains(mh).astype(int)
            df2["medical_history"] = df2.medical_history.str.replace(mh+ ",", "")

        df2["total_diseases"] = df2.medical_history.str.count(",")
        df2 = df2.drop(columns = ["medical_history", "patient_reported_symptoms"])
        return df2

    def load_patient_symptoms(self, df_clean):
        top_symptoms = ['Fever,', 'chills,', 'or sweating,', 'Shortness of breath,',
       'Loss of taste,', 'Loss of smell,', 'New or worsening cough,',
       'Sore throat,', 'Body aches,', 'None,']
        df3 = df_clean.copy()
        for ps in top_symptoms:
            df3[ps] = df_clean.patient_reported_symptoms.str.contains(ps).astype(int)
            df3["patient_reported_symptoms"] = df_clean.patient_reported_symptoms.str.replace(ps+ ",", "")

        df3["total symptoms"] = df_clean.patient_reported_symptoms.str.count(",")
        df3 = df3.drop(columns = ["patient_reported_symptoms", "medical_history", "corona_test", "age", "gender", "smoker"])

        df3.rename(columns={'None,':'Nothing,'}, 
                         inplace=True)
        return df3

    def load_patient_symptoms_test(self, df_clean):
        top_symptoms = ['Fever,', 'chills,', 'or sweating,', 'Shortness of breath,',
       'Loss of taste,', 'Loss of smell,', 'New or worsening cough,',
       'Sore throat,', 'Body aches,', 'None,']
        df3 = df_clean.copy()
        for ps in top_symptoms:
            df3[ps] = df_clean.patient_reported_symptoms.str.contains(ps).astype(int)
            df3["patient_reported_symptoms"] = df_clean.patient_reported_symptoms.str.replace(ps+ ",", "")

        df3["total symptoms"] = df_clean.patient_reported_symptoms.str.count(",")
        df3 = df3.drop(columns = ["patient_reported_symptoms", "medical_history", "age", "gender", "smoker"])

        df3.rename(columns={'None,':'Nothing,'}, 
                         inplace=True)
        return df3

    def merge_df2_df3(self, df2, df3):
        df4 = pd.concat([df2, df3], axis=1, ignore_index=False)
        # here you can drop total symptoms and total diseases column as well, in order to make your algo more robust
        df4 = df4.drop(columns = ["total_diseases", "total symptoms"])
        return df4

    def count_class_per(self, df4):
        ax = sns.countplot(x = "corona_test", data = df4)
        # print(df4["corona_test"].value_counts())
        pos = df4["corona_test"].value_counts()[0]
        neg = df4["corona_test"].value_counts()[1]

        neg_per = 100 *(pos / float(df4.shape[0]))
        pos_per = 100 *(neg / float(df4.shape[0]))
        return (neg_per, pos_per)

    def gen_smok_encoding(self, df4):

        # For checking distribution of values in each column or in each feature.. 
        unique = df4["smoker"].value_counts()
        # print("Number of unique age values :", unique.shape[0])
        
        df4['gender'] = LabelEncoder().fit_transform(df4['gender'])
        df4['smoker'] = LabelEncoder().fit_transform(df4['smoker'])

        return df4

    def gen_smok_encoding_test(self, df4):
        # if df4["smoker"][0] == "no":
        #     df4["smoker"].replace({"no": 0}, inplace=True)
        #     # df4["gender"] = 0
        #     return df4
        # else:
        #     df4["smoker"].replace({"yes": 1}, inplace=True)
        #     # df4["gender"] = 0
        #     return df4
        df4 = df4.drop(columns = ["gender", "smoker"])
        return df4

    def merge_dataframes(self, df4):
        # so here we are merging dataframes together
        df5 = df4.replace(to_replace ="negative", 
                 value =0)
        df5 = df5.replace(to_replace ="positive", 
                 value =1)
        target_labels = df5["corona_test"] 
        df5 = df5.drop(columns = ["corona_test", "gender", "smoker"])
        print("Df5: ", df5.head(1))
        return (df5, target_labels)


    def train_model(self, df5, target_labels, save_model_name):
        X_train, X_test, y_train, y_test = train_test_split(df5, target_labels, test_size=0.30, random_state=20, shuffle=True)
    
        smote = SMOTE(random_state=0)

        X_train_smote , y_train_smote = smote.fit_sample(X_train.astype("int"), y_train.astype("int"))

        # print("Before Smote: ", Counter(y_train))
        # print("After Smote: ", Counter(y_train_smote))

        max_age = X_train_smote["age"].max()
        X_train_smote["age"] = X_train_smote["age"] / max_age
        X_test["age"] = X_test["age"] / max_age
        
        # print("Max age of training patients is", max_age)

        clf = SVC(kernel = "poly", degree = 2, gamma = 10, C = 100,random_state=0, probability=True)
        clf.fit(X_train_smote, y_train_smote)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        print("Accuracy on test dataset : ", accuracy)

        y_test = np.array(y_test).astype("int")
        print(classification_report(y_test, y_pred))
        pd.crosstab(y_test, y_pred)

        filename = save_model_name
        pickle.dump(clf, open(filename, 'wb'))
        return filename
    
    def train_callable(self, csv, save_model_name):
        df_clean = self.load_csv(csv)
        df2 = self.load_medical_history(df_clean)
        df3 = self.load_patient_symptoms(df_clean)
        df4 = self.merge_df2_df3(df2, df3)
        # neg_per, pos_per = self.count_class_per(df4)
        upd_df4 = self.gen_smok_encoding(df4)
        df5, target_labels = self.merge_dataframes(upd_df4)
        filename = self.train_model(df5, target_labels, save_model_name)
        print("Trained model is stored in the file named : ", filename)
        return filename
        

    def norm_test_data(self, data):
        data['age'] = data["age"] / 67
        # print("data normalised: ", data)
        return data

    def test_data(self, csv, model_name):
        # preprocess data
        df_clean = self.load_test_csv(csv)
        df2 = self.load_medical_history(df_clean)
        df3 = self.load_patient_symptoms_test(df_clean)
        df4 = self.merge_df2_df3(df2, df3)
        upd_df4 = self.gen_smok_encoding_test(df4)
        new_data = self.norm_test_data(upd_df4)
        # print(new_data)
        # print(new_data["Pregnancy,"])
        # print(new_data["Diabetes with complications,"])
        # print(new_data["Disease or conditions that make it harder to cough,"])
        # print(new_data["Congestive heart failure,"])
        # print(new_data["Extreme obesity,"])
        # print(new_data["Fever,"])
        # print(new_data["chills,"])
        # print(new_data["or sweating,"])
        # print(new_data["Shortness of breath,"])
        # print(new_data["New or worsening cough,"])
        # print(new_data["Sore throat,"])
        # print(new_data["Body aches,"])
        # print(new_data["Loss of smell,"])
        
        result = self.load(model_name, new_data)
        return result
        
    def load(self, model_name, X_test):
        # load the model from disk
        loaded_model = joblib.load(model_name)
#         loaded_model = pickle.load(open(model_name, 'rb'))
#         result = loaded_model.score(X_test, Y_test)
#         y_pred = loaded_model.predict(X_test_scaled)
#         y_pred = loaded_model.predict_proba(X_test)
        y_pred = loaded_model.predict(X_test)
        #print("tadaaaa the prediction is : ",y_pred)
        return y_pred
