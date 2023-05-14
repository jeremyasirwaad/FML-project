import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from math import log
from re import compile
from urllib.parse import urlparse
from socket import gethostbyname


class LexicalURLFeature:
    def __init__(self, url):
        self.url = url
        self.urlparse = urlparse(self.url)

    def url_length(self):
        return len(self.url)

    def number_of_digits(self):
        return len([i for i in self.url if i.isdigit()])

    def number_of_parameters(self):
        params = self.urlparse.query
        return 0 if params == '' else len(params.split('&'))

    def url_has_port_in_string(self):
        has_port = self.urlparse.netloc.split(':')
        return len(has_port) > 1 and has_port[-1].isdigit()

    def url_path_length(self):
        return len(self.urlparse.path)


pd = pd.read_csv('urldata -Short.csv')

url_length = []
no_of_digits = []
num_of_para = []
has_port = []
url_path_length = []

for index in pd.index:
    a = LexicalURLFeature(pd['url'][index])
    url_length.append(a.url_length())
    no_of_digits.append(a.number_of_digits())
    num_of_para.append(a.number_of_parameters())
    has_port.append(int(a.url_has_port_in_string()))
    url_path_length.append(a.url_path_length())

pd['url_length'] = url_length
pd['no_of_digits'] = no_of_digits
pd['no_of_parameters'] = num_of_para
pd['has_port'] = has_port
pd['url_path_length'] = url_path_length

pd.to_csv('features.csv', index=False)

le = LabelEncoder()
pd['label'] = le.fit_transform(pd['label'])

X = pd.drop(['label', 'url'], axis=1)
y = pd['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


def predict_url(url):
    a = LexicalURLFeature(url)
    url_length = a.url_length()
    no_of_digits = a.number_of_digits()
    num_of_para = a.number_of_parameters()
    has_port = int(a.url_has_port_in_string())
    url_path_length = a.url_path_length()
    label_feature = pd['label'][0]
    features = [url_length, no_of_digits, num_of_para,
                has_port, url_path_length, label_feature]
    features = scaler.transform([features])

    label_pred = knn.predict(features)
    label_pred = le.inverse_transform(label_pred)

    return label_pred[0]


url = input("Enter a URL to predict: ")
if not url:
    print(f"The URL {url} is predicted as {predict_url(url)}")
