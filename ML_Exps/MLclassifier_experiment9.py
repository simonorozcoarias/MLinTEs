import matplotlib as mpl
mpl.use('Agg')
import seaborn as sn; sn.set()
import pandas as pd
import numpy as np
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
    #schemes = ["EIIP"]#, "Galois4","orthogonal", "EIIP"]# este es para SVM que falta
    #schemes = ["Galois4"]
    schemes = ["kmers"]#, "pc", "complementary", "DAX", "EIIP", "enthalpy", "Galois4", "orthogonal"]
    #schemes = ["complementary", "DAX", "EIIP", "enthalpy", "Galois4", "orthogonal"]
    # evaluate each model in turn
    results = []
    names = []
    for scheme in schemes:

        training_data = pd.read_csv(filename+'.'+scheme, index_col=False)
        print(training_data)

        # basic statistics
        #training_data.describe()
        label_vectors =training_data['Label'].values
        #label_vectors = [int(x) for x in label_vectors]
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
        results = []
        names = []
        seed = 7
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        X_train, X_validation, Y_train, Y_validation = train_test_split(x_data, list(y_data), test_size=validation_size, random_state=seed)
        x_data = None
        y_data = None
        training_data = None
        label_vectors = None
        feature_vectors = None

        # Logistic Regression
        # Testing different values of C
        limit=1
        step=0.1
        x=[0 for x in range(0,int(limit/step))]
        yValidation=[0 for x in range(0,int(limit/step))]
        ytrain=[0 for x in range(0,int(limit/step))]
        i=step
        index=0
        while i<limit:
            lr = LogisticRegression(C=i, n_jobs=threads)
            lr.fit(X_train,Y_train)
            trainScore=f1_score(Y_train, lr.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, lr.predict(X_validation), average='macro')
            ytrain[index]=trainScore
            yValidation[index]=validationScore
            print('ite:',i)
            x[index]=i
            i+=step
            index+=1

        lr = LogisticRegression(C=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(lr, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('LR')
        msg = "%s: %f (%f) with C: %s" % ('### LR', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,LR,'+','.join([str(x) for x in cv_results.tolist()])) 
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
        
        LDA = LinearDiscriminantAnalysis(tol=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(LDA, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('LDA')
        msg = "%s: %f (%f) with tol: %s" % ('### LDA', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,LDA,'+','.join(cv_results.tolist()))
        
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
       
        KNN = KNeighborsClassifier(n_neighbors=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(KNN, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('KNN')
        msg = "%s: %f (%f) with Neighbors: %s" % ('### KNN', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,KNN,'+','.join(cv_results))

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

        MLP = MLPClassifier(solver='lbfgs', alpha=.5, hidden_layer_sizes=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(MLP, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('MLP')
        msg = "%s: %f (%f) with Neurons: %s" % ('### MLP', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,MLP,'+','.join(cv_results))

        #free memory
        MLP = None

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
	
        RF = RandomForestClassifier(n_estimators=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(RF, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('RF')
        msg = "%s: %f (%f) with n_estimators: %s" % ('### RF', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,RF,'+','.join(cv_results))

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
        
        DT = DecisionTreeClassifier(max_depth=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(DT, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('DT')
        msg = "%s: %f (%f) with max_depth: %s" % ('### DT', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,DT,'+','.join(cv_results))        

        #free memory
        DT = None

	    # Bayesian Classifier
        limit=1e-15
        step=1e-2
        values=[1e-1,1e-3,1e-5,1e-7,1e-9,1e-11,1e-13,1e-15,1e-17,1e-19]
        x=[0 for x in range(0,len(values))]
        yValidation=[0 for x in range(0,len(values))]
        ytrain=[0 for x in range(0,len(values))]
        for i,index in zip(values,range(len(values))):
            NB = GaussianNB(var_smoothing=i)

            NB.fit(X_train,Y_train)
            trainScore=f1_score(Y_train, NB.predict(X_train), average='macro')
            validationScore=f1_score(Y_validation, NB.predict(X_validation), average='macro')

            ytrain[index]=trainScore
            yValidation[index]=validationScore

            print('ite:',i)

            x[index]=i
	
        NB = GaussianNB(var_smoothing=x[yValidation.index(max(yValidation))])
        cv_results = model_selection.cross_val_score(NB, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('NB')
        msg = "%s: %f (%f) with var_smoothing: %s" % ('### NB', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,NB,'+','.join(cv_results))

        #free memory
        NB = None

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

        svc = SVC(C=x[yValidation.index(max(yValidation))],gamma=1e-6)
        cv_results = model_selection.cross_val_score(svc, X_train,Y_train, cv=kfold, scoring='f1_macro')
        results.append(cv_results)
        names.append('svc')
        msg = "%s: %f (%f) with C: %s" % ('### svc', cv_results.mean(), cv_results.std(), str(x[yValidation.index(max(yValidation))]))
        print(msg)
        #print('###,SVC,'+','.join(cv_results))

        #releasing memory
        svc = None

        # Compare Algorithms Accuracy
        plt.close('all')
        fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        fig.suptitle('Algorithm Comparison')     
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig('comparisons.png', dpi=300)

        #print(results)
        #print(+names)

        # print results
        for i in range(len(results)):
        	print('###|'+names[i]+'|'+'|'.join([str(x) for x in results[i].tolist()]))



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
    #classification(fileData, 1, threads)
    #classification(fileData, 2, threads)
    #classification(fileData, 3, threads)
    classification(fileData, 4, threads)
