import math
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from math import log
from re import compile
from urllib.parse import urlparse
from socket import gethostbyname
from tldextract import extract


class LexicalURLFeature:
    def __init__(self, url):
        self.url = url
        self.urlparse = urlparse(self.url)
        self.extract = extract(self.url)

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

    def is_https(self):
        return int(self.urlparse.scheme == "https")

    def no_of_sub_domains(self):
        return len(self.extract.subdomain.split('.'))

    def url_entropy(self):
        s = list(self.url)
        probabilities = [s.count(i)/len(s) for i in set(s)]
        entropy = -sum([p*math.log2(p) for p in probabilities])
        return entropy


    def no_of_special_chars(self):
        return len(re.findall('[^A-Za-z0-9]', self.url))

    def contains_IP(self):
        return int(bool(re.match("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", self.url)))

    def no_of_subdir(self):
        return self.url.count('/')

    def url_is_encoded(self):
        return int(self.url != re.sub('%[0-9a-fA-F]{2}', '', self.url))

    def domain_length(self):
        return len(self.extract.domain)

    def no_of_queries(self):
        return len(self.urlparse.query.split('&'))

    def avg_token_length(self):
        tokens = re.split('[/-]', self.url)
        avg_length = sum(len(token) for token in tokens)/len(tokens)
        return avg_length

    def token_count(self):
        tokens = re.split('[/-]', self.url)
        return len(tokens)

    def largest_token(self):
        tokens = re.split('[/-]', self.url)
        return len(max(tokens, key=len))

    def smallest_token(self):
        tokens = re.split('[/-]', self.url)
        return len(min(tokens, key=len))

    def contains_at_symbol(self):
        return int('@' in self.url)

    def is_shortened(self):
        shortening_services = ["bit.ly", "t.co", "goo.gl", "tinyurl.com", "tr.im", "is.gd", "cli.gs",
                               "yfrog.com", "migre.me", "ff.im", "tiny.cc", "url4.eu", "twit.ac",
                               "su.pr", "twurl.nl", "snipurl.com", "short.to", "budurl.com", "ping.fm",
                               "post.ly", "just.as", "bkite.com", "snipr.com", "fic.kr", "loopt.us",
                               "doiop.com", "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly",
                               "to.ly", "bit.do", "t.co", "lnkd.in", "db.tt", "qr.ae", "adf.ly",
                               "goo.gl", "bitly.com", "cur.lv", "tinyurl.com", "ow.ly", "bit.ly",
                               "adcrun.ch", "zpag.es", "ity.im", "q.gs", "is.gd", "po.st", "bc.vc",
                               "twitthis.com", "u.to", "j.mp", "buzurl.com", "cutt.us", "u.bb",
                               "yourls.org", "x.co", "prettylinkpro.com", "scrnch.me", "filoops.info",
                               "vzturl.com", "qr.net", "1url.com", "tweez.me", "v.gd", "tr.im",
                               "link.zip.net"]

        domain = self.extract.domain + '.' + self.extract.suffix
        return int(domain in shortening_services)

    def count_dots(self):
        return self.url.count('.')

    def count_delimiters(self):
        delimiters = [';', '_', '?', '=', '&']
        return sum([self.url.count(d) for d in delimiters])

    def count_sub_domains(self):
        return self.url.count('.')

    def is_www(self):
        return int(self.extract.subdomain == "www")

    def count_reserved_chars(self):
        reserved_chars = [';', '/', '?', ':', '@', '&', '=', '+', '$', ',']
        return sum([self.url.count(c) for c in reserved_chars])


pd = pd.read_csv('urldata -Short.csv')

url_length = []
no_of_digits = []
num_of_para = []
has_port = []
url_path_length = []
is_https = []
no_of_sub_domains = []
url_entropy = []
no_of_special_chars = []
contains_IP = []
no_of_subdir = []
url_is_encoded = []
domain_length = []
no_of_queries = []
avg_token_length = []
token_count = []
largest_token = []
smallest_token = []
contains_at_symbol = []
is_shortened = []
count_dots = []
count_delimiters = []
count_sub_domains = []
is_www = []
count_reserved_chars = []

for index in pd.index:
    a = LexicalURLFeature(pd['url'][index])
    url_length.append(a.url_length())
    no_of_digits.append(a.number_of_digits())
    num_of_para.append(a.number_of_parameters())
    has_port.append(int(a.url_has_port_in_string()))
    url_path_length.append(a.url_path_length())
    is_https.append(a.is_https())
    no_of_sub_domains.append(a.no_of_sub_domains())
    url_entropy.append(a.url_entropy())
    no_of_special_chars.append(a.no_of_special_chars())
    contains_IP.append(a.contains_IP())
    no_of_subdir.append(a.no_of_subdir())
    url_is_encoded.append(a.url_is_encoded())
    domain_length.append(a.domain_length())
    no_of_queries.append(a.no_of_queries())
    avg_token_length.append(a.avg_token_length())
    token_count.append(a.token_count())
    largest_token.append(a.largest_token())
    smallest_token.append(a.smallest_token())
    contains_at_symbol.append(a.contains_at_symbol())
    is_shortened.append(a.is_shortened())
    count_dots.append(a.count_dots())
    count_delimiters.append(a.count_delimiters())
    count_sub_domains.append(a.count_sub_domains())
    is_www.append(a.is_www())
    count_reserved_chars.append(a.count_reserved_chars())

pd['url_length'] = url_length
pd['no_of_digits'] = no_of_digits
pd['no_of_parameters'] = num_of_para
pd['has_port'] = has_port
pd['url_path_length'] = url_path_length
pd['is_https'] = is_https
pd['no_of_sub_domains'] = no_of_sub_domains
pd['url_entropy'] = url_entropy
pd['no_of_special_chars'] = no_of_special_chars
pd['contains_IP'] = contains_IP
pd['no_of_subdir'] = no_of_subdir
pd['url_is_encoded'] = url_is_encoded
pd['domain_length'] = domain_length
pd['no_of_queries'] = no_of_queries
pd['avg_token_length'] = avg_token_length
pd['token_count'] = token_count
pd['largest_token'] = largest_token
pd['smallest_token'] = smallest_token
pd['contains_at_symbol'] = contains_at_symbol
pd['is_shortened'] = is_shortened
pd['count_dots'] = count_dots
pd['count_delimiters'] = count_delimiters
pd['count_sub_domains'] = count_sub_domains
pd['is_www'] = is_www
pd['count_reserved_chars'] = count_reserved_chars

pd.to_csv('features.csv', index=False)

le = LabelEncoder()
pd['label'] = le.fit_transform(pd['label'])

feature_columns = ['url_length', 'no_of_digits', 'no_of_parameters', 'has_port', 
                   'url_path_length', 'is_https', 'no_of_sub_domains', 
                   'url_entropy', 'no_of_special_chars', 'contains_IP', 
                   'no_of_subdir', 'url_is_encoded', 'domain_length', 
                   'no_of_queries', 'avg_token_length', 'token_count', 
                   'largest_token', 'smallest_token', 'contains_at_symbol', 
                   'is_shortened', 'count_dots', 'count_delimiters', 
                   'count_sub_domains', 'is_www', 'count_reserved_chars']

X = pd[feature_columns]

y = pd['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def predict_url(url):
    a = LexicalURLFeature(url)
    features = [
        a.url_length(),
        a.number_of_digits(),
        a.number_of_parameters(),
        int(a.url_has_port_in_string()),
        a.url_path_length(),
        a.is_https(),
        a.no_of_sub_domains(),
        a.url_entropy(),
        a.no_of_special_chars(),
        a.contains_IP(),
        a.no_of_subdir(),
        a.url_is_encoded(),
        a.domain_length(),
        a.no_of_queries(),
        a.avg_token_length(),
        a.token_count(),
        a.largest_token(),
        a.smallest_token(),
        a.contains_at_symbol(),
        a.is_shortened(),
        a.count_dots(),
        a.count_delimiters(),
        a.count_sub_domains(),
        a.is_www(),
        a.count_reserved_chars(),
    ]
    features = scaler.transform([features])

    label_pred = knn.predict(features)
    label_pred = le.inverse_transform(label_pred)

    return label_pred[0]


url = input("Enter a URL to predict: ")
if not url:
    url = "https://goole.com.xyz"
print(f"The URL {url} is predicted as {predict_url(url)}")
