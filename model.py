import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data.csv")

X = df[["Height", "Weight"]]
y = df["Species"]

clf = GaussianNB() 
clf.fit(X, y)

joblib.dump(clf, "clf.pkl")
