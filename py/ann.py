import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def create_dataset(n=8, samples=10000):
    l = list()
    for i in range(samples):
        b = np.random.randint(0, 1000, size=(n, n))
        b_symm = (b + b.T) / 2
        for i in range(n):
            b_symm[i][i] = 0
        b_symm = b_symm.flatten()
        l.append(b_symm.tolist())

    dataset = np.array(l, np.float64)

    outcome = [0 for i in range(samples)]

    for s in range(samples):
        tmp = dataset[s]
        tmp = tmp.reshape(n, n)
        for i in range(n):
            for j in range(n):
                if np.random.randint(0, 10) > 5:
                    tmp[i][j] = tmp[i][j] + np.random.randint(0, 20)
        for i in range(n):
            tmp[i][i] = 0
        for i in range(n):
            for j in range(n):
                if np.random.randint(0, 10) > 6:
                    tmp[i][j] = 0
                    tmp[j][i] = 0
        if s % 2 == 0:
            for i in range(n):
                for j in range(n):
                    if np.random.randint(0, 10) > 8 and i != j:
                        tmp[i][j] = tmp[i][j] + np.random.randint(200, 300)
            outcome[s] = 1
        dataset[s] = tmp.flatten()
    return dataset, outcome


X, y = create_dataset()

print X[0].reshape(8, 8)
print y

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(64, 64, 64), verbose=True)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
