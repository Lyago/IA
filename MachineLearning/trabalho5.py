import arff, numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
#Supremir DeprecationWarnings
warnings.filterwarnings("ignore")


dataset = arff.load(open('spambase.arff', 'rb'))
data = np.array(dataset['data'])
data = data.astype(np.float64)

#dados sem coluna final
X = data[::, :-1:]

#coluna final, onde se classifica se o email e spam ou nao
y = data[::,-1::]

#primeira divisao de dados: conjunto D e conjunto de testes
data_D, data_test, y_D, y_test = train_test_split(X, y, test_size=0.2)
#segunda divisao: conjunto de treinamento e conjunto de validacao a partir do conjunto D
data_train, data_validation, y_train, y_validation = train_test_split(data_D, y_D, test_size=0.2)


# obtendo o melhor k
k_range = list(range(1,26))
scores = []
for k in k_range:
    #gerando modelo de previsao baseado no conjunto de treinamento
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_train, y_train)

    # testando acuracia no conjunto de validacao
    y_pred = knn.predict(data_validation)
    scores.append(metrics.accuracy_score(y_validation, y_pred))
#resultados
print ("accuracy per k: ")
print(scores)
best_k = np.argmax(scores)+1
print ("Best k: ")
print(best_k)


knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(data_D, y_D)
y_pred = knn.predict(data_test)
print("KNN's accuracy:")
print(metrics.accuracy_score(y_test, y_pred))


logreg = LogisticRegression()
logreg.fit(data_D, y_D)
y_pred = logreg.predict(data_test)
print("Logistic Regression's accuracy:")
print(metrics.accuracy_score(y_test, y_pred))



