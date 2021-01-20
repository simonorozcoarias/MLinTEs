#!/usr/bin/env python
# coding: utf-8

# # Ensamble de métodos

# ### Librerías

# In[10]:


import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas import set_option

from sklearn import decomposition, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

import random
seed = random.seed(7)
import time
Fstart_t = time.time()


# ### Funciones escenciales

# In[2]:


def metrics(Y_validation,predictions):
    '''Esta función permite calcular las métricas de los modelos entrenados'''
    #print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    #print('\n clasification report:\n', classification_report(Y_validation, predictions))
    #print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    snn_cm = confusion_matrix(Y_validation, predictions)
    return snn_cm

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    '''Esta función es para la gráfica de la curva de aprendizaje de los modelos'''
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt    


# ### Configuraciones iniciales del script

# In[3]:


set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None
scoring = 'f1_weighted'
models = []
results = []
names = []
filename = "/home/bioml/Projects/PhD/InpactorDB/InpactorDB_non-redundant+negative.fasta.508_kmers"
n_jobs = 64

# ### Carga de la base de datos

# In[5]:


training_data = pd.read_csv(filename)
#training_data = training_data[:100]  # Esta linea es debido a poca RAM y pruebas. Borrarlas en server
label_vectors = training_data['Label'].values
feature_vectors = training_data.drop(['Label'],axis=1)
x_data = []
y_data = []


# ### Aplicamos el esquema de procesamiento ya definido Scaling + PCA

# In[6]:


# information scaling
scaler = StandardScaler().fit(feature_vectors)
feature_vectors_scaler = scaler.transform(feature_vectors)

# PCA with scaling
pca = PCA(n_components=0.9,svd_solver='full',tol=1e-4)
pca.fit(feature_vectors_scaler)
X_trainPCAScaler = pca.transform(feature_vectors_scaler)
x_data = X_trainPCAScaler
y_data = label_vectors


# ### División del dataset

# In[8]:


training_size = .90
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size=training_size, random_state=seed)

print("Y_train length = ", Y_train.shape[0])
print("Y_test length = ", Y_test.shape[0])

new_perc = (Y_test.shape[0]*100)/Y_train.shape[0]
print("New percentage is= {:.2f}%".format(new_perc))
# 1/CV is the percentage if data to test the model with cross validation, 
# taken from:https://www.statisticshowto.com/cross-validation-statistics/
CV = round(1/(new_perc/100))
print("K-fold is equal to ",CV)

Xtraint, Xval, Ytraint, Yval = train_test_split(X_train,Y_train,
                                                test_size=new_perc/100, random_state=seed)


# ### Ensambling method

# In[20]:


clf1 = KNeighborsClassifier(algorithm='auto', n_neighbors=2, weights='distance', metric = 'euclidean')
clf2 = LogisticRegression(C=0.01, tol=10, max_iter=1000, penalty='l2',solver='sag')
clf3 = LinearDiscriminantAnalysis(shrinkage=0.0001,solver='lsqr',tol=0.1)

Rf = RandomForestClassifier()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=Rf)
text_model = "Stacking"


# In[21]:


label = ['KNN','LR', 'LDA','Stacking Classifier']
clf_list = [clf1, clf2, clf3, sclf]

clf_cv_mean = []
clf_cv_std = []
for clf, label in zip(clf_list, label):
        
    scores = cross_val_score(clf, X_train,Y_train, cv=CV, scoring=scoring, n_jobs=n_jobs)
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std()) 


# #### Final Model evaluation 

# In[13]:


Cros_val = cross_val_score(sclf,X_train,Y_train,cv=CV,scoring=scoring, n_jobs=n_jobs)
print("Cross validation score: ",np.asarray(Cros_val).mean())

sclf.fit(Xtraint,Ytraint)
prediction = sclf.predict(X_test)

snn_cm = metrics(Y_test,prediction)
print("#####################################")
# Visualizamos la matriz de confusión
snn_df_cm = pd.DataFrame(snn_cm)  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 25}) # font size
plt.savefig(str(text_model)+"_Confussion_matrix.png", dpi=300)

fig, axes = plt.subplots(3, 1, figsize=(10, 15))
title = "Learning Curves (tunned "+str(text_model)+")"

plot_learning_curve(sclf, title, X_train, Y_train,
                    cv=CV, n_jobs=n_jobs)
#plt.show()
plt.savefig(str(text_model)+"_Tunned_Algorithm.png", dpi=300)
Fend_t = time.time()
print("Full Script Tunning time",Fend_t-Fstart_t)
