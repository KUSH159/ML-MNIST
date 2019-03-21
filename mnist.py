from sklearn.ensemble import RandomForestClassifier as RFC
import pandas as pd
import numpy as np
import csv

train = pd.read_csv('train.csv')
train_labels = train['label'].values.flatten()

train = train.drop(['label'], axis = 1)

test = pd.read_csv('test.csv')

clf = RFC(n_estimators=501)
clf.fit(train, train_labels)

pred = clf.predict(test)
print (np.shape(pred))
writer = csv.writer(open("result.csv", "w"))
head = ["ImageId", "Label"]
writer.writerows([head])
cnt = 1
for row in pred:
    writer.writerows([[str(cnt), str(int(row))]])
    cnt = cnt+1