import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"\n✅ Training Complete!")
print(f"Overall Accuracy: {score * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_predict, zero_division=0))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
