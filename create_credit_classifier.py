import pandas as pd

#Load dataset
df = pd.read_csv('cs-training.csv',sep=',',header=0)
data = df.drop(df.columns[0], axis=1)

#Drop rows with missing column data
data = data.dropna()

# Convert data into list of dict records
data = data.to_dict(orient='records')

#Seperate Target and outcome features
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame

vec = DictVectorizer()

df_data = vec.fit_transform(data).toarray()
feature_names = vec.get_feature_names()
df_data = DataFrame(
    df_data,
    columns = feature_names)

outcome_feature = df_data['SeriousDlqin2yrs']
target_features = df_data.drop('SeriousDlqin2yrs', axis=1)

from sklearn import cross_validation
X_1, X_2, Y_1, Y_2 = cross_validation.train_test_split(
    target_features, outcome_feature, test_size=0.5, random_state=0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_1,Y_1)
output = clf.predict(X_2)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(output, Y_2)
score = clf.score(X_2, Y_2)
print "accuracy: {0}".format(score.mean())
print matrix


###
### Save Classifier
###
from sklearn.externals import joblib
joblib.dump(clf, 'model/nb.pkl')
