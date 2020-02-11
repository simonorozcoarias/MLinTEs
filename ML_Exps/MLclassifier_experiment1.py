import seaborn as sn; sn.set()
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option


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

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode

"""
This experiment uses all ML algorithms to test classification using different coding schemes and 
different pre-process techniques, but using Cross Validation and default parameters
"""
def classification(filename, prepro):
    set_option("display.max_rows", 10)
    pd.options.mode.chained_assignment = None

    # Cross Validation
    seed = 42
    scoring = 'accuracy'

    models = []
    schemes = ["complementary", "DAX", "EIIP", "enthalpy", "Galois4", "orthogonal", "kmers", "pc"]
    # schemes = ["kmers", "pc"]
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('MLP', MLPClassifier()))
    models.append(('KM', KMeans()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
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
            feature_vectors_scaler = scaler.transform(feature_vectors)
            x_data = feature_vectors_scaler
            y_data = label_vectors
            print("### Scaling")
        elif prepro == 3:
            # PCA without scaling
            pca = decomposition.PCA(n_components=0.96,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors)
            X_trainPCA = pca.transform(feature_vectors)
            x_data = X_trainPCA
            y_data = label_vectors
            print("### PCA")
            print('X_PCA:', X_trainPCA.shape)
        elif prepro == 4:
            # information scaling
            scaler = preprocessing.StandardScaler().fit(feature_vectors)
            feature_vectors_scaler = scaler.transform(feature_vectors)
            validation_size = 0.2
            seed = 7
            
            # PCA with scaling
            pca = decomposition.PCA(n_components=0.9,svd_solver='full',tol=1e-4)
            pca.fit(feature_vectors_scaler)
            X_trainPCAScaler = pca.transform(feature_vectors_scaler)
            x_data = X_trainPCAScaler
            y_data = label_vectors
            print("### PCA + Scaling")
            print('X_PCA:',X_trainPCAScaler.shape)

        msg = ""
        for name, model in models:
            print("doing "+name)
            # segmentar la base de datos de entrenamiento en 10 partes
            kfold = model_selection.KFold(n_splits=5, random_state=seed)
            # probar los grupos en cada modelo
            cv_results = model_selection.cross_val_score(model, x_data, y_data, cv=kfold, scoring=scoring)
            # almacena resultados
            results.append(cv_results)
            names.append(name)
            msg += " %f +- %f" % (cv_results.mean(), cv_results.std())+", "
        print("### results, "+scheme+","+msg)

def metrics(Y_validation,predictions):
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #Creamos la matriz de confusión
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusión
    snn_df_cm = pd.DataFrame(snn_cm, range(11), range(11))
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()

if __name__ == '__main__':
    fileData = "/home/orozco/Doctorado/databasesForML/finalDatabases/lineages_classification/Repbase/repbase_LTRs_I_3dom.fasta.lineages_final"
    classification(fileData, 1) # for any pre-procesing
    classification(fileData, 2) # for Scaling
    classification(fileData, 3) # for PCA
    classification(fileData, 4) # for Scaling + PCA

