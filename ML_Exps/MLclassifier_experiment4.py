import seaborn as sn; sn.set()
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option

import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode

def classification(filename, prepro, threads):
    set_option("display.max_rows", 10)
    pd.options.mode.chained_assignment = None

    # Cross Validation
    seed = 42
    scoring = 'accuracy'

    models = []
    #schemes = ["DAX"]
    #schemes = ["complementary", "DAX", "EIIP", "enthalpy", "Galois4", "orthogonal"]
    schemes = ["pc"]#["kmers", "pc", "complementary", "DAX", "EIIP", "enthalpy", "Galois4", "orthogonal"]
    # evaluate each model in turn
    results = []
    names = []
    for scheme in schemes:

        training_data = pd.read_csv(filename+'.'+scheme)
        print(training_data)

        # basic statistics
        training_data.describe()
        label_vectors = training_data['Label'].values
        feature_vectors = training_data.drop(['Label'], axis=1).values
        print(label_vectors)
        print(feature_vectors)

        x_data = []
        y_data = []

        if prepro == 1:
            x_data = feature_vectors
            y_data = label_vectors
            print("### Any")
        elif prepro == 2:
            # information scaling
            scaler = preprocessing.StandardScaler().fit(feature_vectors)
            x_data = scaler.transform(feature_vectors)
            y_data = label_vectors
            print("### Scaling")
        elif prepro == 3:
            # PCA without scaling
            pca = decomposition.PCA(n_components=0.96,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors)
            x_data = pca.transform(feature_vectors)
            y_data = label_vectors

            #releasing memory
            pca = None
            label_vectors = None
            print("### PCA")
            print('X_PCA:', x_data.shape)
        elif prepro == 4:
            # information scaling
            scaler = preprocessing.StandardScaler().fit(feature_vectors)
            feature_vectors_scaler = scaler.transform(feature_vectors)
            
            # PCA with scaling
            pca = decomposition.PCA(n_components=0.9,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors_scaler)
            x_data = pca.transform(feature_vectors_scaler)
            y_data = label_vectors

            # realeasing memory
            feature_vectors_scaler = None
            pca = None
            scaler = None
            print("### PCA + Scaling")
            print('X_PCA:',x_data.shape)

        # information split with scaling
        validation_size = 0.2
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(x_data, y_data, test_size=validation_size, random_state=seed)

        # Logistic Regression
        # Testing different values of C
        """limit=1
        step=0.1
        x=[0 for x in range(0,int(limit/step))]
        yValidation=[0 for x in range(0,int(limit/step))]
        ytrain=[0 for x in range(0,int(limit/step))]
        i=step
        index=0
        while i<limit:
            lr = LogisticRegression(C=i, n_jobs=threads)

            lr.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, lr.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, lr.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
            i+=step
            index+=1
	
        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        
        plt.xlabel('C')
        plt.ylabel('F1-Score')
        plt.ylim((0,1.1))
        #plt.title('C vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('LR-Algorithm_'+scheme+'.png', dpi=100)
        print('### '+scheme)
        print('### LG ### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
        lr = LogisticRegression(C=x[yValidation.index(max(yValidation))])
        lr.fit(X_train, Y_train)
        predictions = lr.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        lr = None

        # LDA
        # Testing different values of tolerance
        limit=0.001
        step=0.0001
        x=[0 for x in range(0,int(limit/step))]
        yValidation=[0 for x in range(0,int(limit/step))]
        ytrain=[0 for x in range(0,int(limit/step))]
        i=step
        index=0
        while i<=limit:
            LDA = LinearDiscriminantAnalysis(tol=i)
            LDA.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, LDA.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, LDA.predict(X_validation), average='macro')
            ytrain[index]=trainScore
            yValidation[index]=validationScore
            print('ite:',i)

            x[index]=i
            i=round(i+step,4)
            index+=1

        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        plt.ylim((0,1.1))
        plt.xlabel('C')
        plt.ylabel('F1-Score')
        #plt.title('Tolerance vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('LDA-Algorithm_'+scheme+'.png', dpi=100)
        print('### '+scheme)
        print('### LDA ### The best score with data validation: ', max(yValidation),'with tol: ',x[yValidation.index(max(yValidation))])
        
        LDA = LinearDiscriminantAnalysis(tol=x[yValidation.index(max(yValidation))])
        LDA.fit(X_train, Y_train)
        predictions = LDA.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        LDA = None

        # KNN algorithm
        # Testing different quantities of neighbors
        limit=100
        x=[x for x in range(1,limit, 10)]
        yValidation=[0 for x in range(1,limit, 10)]
        ytrain=[0 for x in range(1,limit, 10)]
        index = 0
        for i in range(1,limit, 10):
            KNN = KNeighborsClassifier(n_neighbors=i, n_jobs=threads)

            KNN.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, KNN.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, KNN.predict(X_validation), average='macro')

            print('KNN Score:',i)
            ytrain[index]=trainScore
            yValidation[index]=validationScore
            index += 1

        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,label='Train')
        plt.ylim((0,1.1))
        plt.plot(x,yValidation,label='Validation')
        plt.xlabel('n-Neighbors')
        plt.ylabel('F1-Score')
        #plt.title('Neighbors vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('KNN-Algorithm_'+scheme+'.png', dpi=100)

        print('### '+scheme)
        print('### KNN ### The best score with data validation: ', max(yValidation),'with Neighbors: ',x[yValidation.index(max(yValidation))])
        
        KNN = KNeighborsClassifier(n_neighbors=x[yValidation.index(max(yValidation))])
        KNN.fit(X_train, Y_train)
        predictions = KNN.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        KNN = None
              
        # MLP
        limit=500
        step=50
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=(i))
            MLP.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, MLP.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, MLP.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('it:',i)
            x[index]=i
            i+=step
            index+=1
	
        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        plt.ylim((0,1.1))
        plt.xlabel('Neurons')
        plt.ylabel('F1-Score')
        #plt.title('Neurons vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('MLP-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### MLP ### The best score with data validation: ', max(yValidation),'with Neurons: ',x[yValidation.index(max(yValidation))])
        MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=x[yValidation.index(max(yValidation))])
        MLP.fit(X_train, Y_train)
        predictions = MLP.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        MLP = None

        # KMeans
        limit=13
        step=1
        i=step
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        
        index=0
        while i<limit:
            KM = KMeans(n_clusters=i, random_state=0, n_jobs=threads)
            clusters =KM.fit_predict(X_train)
            labels = np.zeros_like(clusters)
            for j in range(12):
                mask = (clusters == j)
                labels[mask] = mode(Y_train[mask])[0]
            validationScore=f1_score(Y_train, labels, average='macro')
         
            yValidation[index]=validationScore

            print('it:',i)
            x[index]=i
            i+=step
            index+=1
	
        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,yValidation,'-',label='Validation')
        
        plt.xlabel('Clusters')
        plt.ylabel('F1-Score')
        plt.ylim((0,1.1))
        #plt.title('Clusters vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('Clusters-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### Kmeans ### The best score with data validation: ', max(yValidation),'with Clusters: ',x[yValidation.index(max(yValidation))])
        
        KM = KMeans(n_clusters=x[yValidation.index(max(yValidation))], random_state=0)
        clusters =KM.fit_predict(X_train)
        labels = np.zeros_like(clusters)
        for j in range(12):
            mask = (clusters == j)
            labels[mask] = mode(Y_train[mask])[0]
        metrics(Y_train, labels)
        #free memory
        KM = None

        # RF
        limit=100
        step=10
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            RF = RandomForestClassifier(n_estimators=i, n_jobs=threads)
            RF.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, RF.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, RF.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('n_estimators:',i)
            x[index]=i
            i+=step
            index+=1
	
        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        plt.ylim((0,1.1))
        plt.xlabel('Trees')
        plt.ylabel('F1-Score')
        #plt.title('Trees vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('RF-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### RF ### The best score with data validation: ', max(yValidation),'with n_estimators: ',x[yValidation.index(max(yValidation))])
        RF = RandomForestClassifier(n_estimators=x[yValidation.index(max(yValidation))])
        RF.fit(X_train, Y_train)
        predictions = RF.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        RF = None

        # DT
        limit=10
        step=1
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            DT = DecisionTreeClassifier(max_depth=i)
            DT.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, DT.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, DT.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('max_depth:',i)
            x[index]=i
            i+=step
            index+=1

        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        plt.ylim((0,1.1))
        plt.xlabel('Max Depth')
        plt.ylabel('F1-Score')
        #plt.title('Max Depth vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('DT-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### DT ### The best score with data validation: ', max(yValidation),'with max_depth: ',x[yValidation.index(max(yValidation))])
        DT = DecisionTreeClassifier(max_depth=x[yValidation.index(max(yValidation))])
        DT.fit(X_train, Y_train)
        predictions = DT.predict(X_validation)
        metrics(Y_validation,predictions)
        #free memory
        DT = None"""

        # SVC
        #Testing different values of C
        limit=100
        step=10
        x=[x for x in range(0,int(limit/step)-1)]
        yValidation=[0 for x in range(0,int(limit/step)-1)]
        ytrain=[0 for x in range(0,int(limit/step)-1)]
        
        i=step
        index=0
        while i<limit:
            svc = OneVsRestClassifier(SVC(C=i, gamma=1e-6), n_jobs = threads)

            svc.fit(X_train, Y_train)
            trainScore=f1_score(Y_train, svc.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, svc.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
            i+=step
            index+=1
	
        plt.close('all')
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x,ytrain,'-',label='Train')
        plt.plot(x,yValidation,'-',label='Validation')
        plt.ylim((0,1.1))
        plt.xlabel('C')
        plt.ylabel('F1-Score')
        #plt.title('C vs Accuracy ('+scheme+')')
        plt.legend()
        plt.savefig('SVC-Algorithm_'+scheme+'.png', dpi=100)
        plt.show()
        print('### '+scheme)
        print('### SVM ### The best score with data validation: ', max(yValidation),'with C: ',x[yValidation.index(max(yValidation))])
        
        svc = SVC(C=x[yValidation.index(max(yValidation))],gamma=1e-6)
        svc.fit(X_train, Y_train)
        predictions = svc.predict(X_validation)
        metrics(Y_validation,predictions)
        #releasing memory
        svc = None


def metrics(Y_validation,predictions):
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
    print('Recall:', recall_score(Y_validation, predictions, average='macro'))
    print('Precision:', precision_score(Y_validation, predictions, average='macro'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #Creamos la matriz de confusión
    # snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusión
    # snn_df_cm = pd.DataFrame(snn_cm, range(12), range(12))
    # plt.figure(figsize = (20,14))
    # sn.set(font_scale=1.4) #for label size
    # sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    # plt.show()

if __name__ == '__main__':

    #fileData = "/home/orozco/Doctorado/databasesForML/finalDatabases/lineages_classification/Repbase/repbase_LTRs_I_3dom.fasta.lineages_final"
    #threads=20
    fileData=sys.argv[1]
    threads=int(sys.argv[2])
    print(fileData)
    print(threads)
    classification(fileData, 2, threads)
    #classification(fileData, 2, threads)
    #classification(fileData, 3, threads)
    #classification(fileData, 4, threads)

