import pandas as pd
import utils
from sklearn import linear_model

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "SibSp", "Parch"]].values

classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier.fit(features, target).score(features, target))
