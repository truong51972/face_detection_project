import pickle
import numpy as np 

from pathlib import Path
from sklearn.svm import SVC

class Classifier_model:
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            self.model = SVC(probability=True)
        else:
            with open(path, 'rb') as file:
                self.model = pickle.load(file)

    def train(self, df):
        try:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            self.model.fit(X, y)
        except ValueError:
            print('Need more than 1 people to classify!')
    def save(self, model_path: str = './model', model_name: str = 'svm_model.pkl'):
        path = Path(model_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / model_name, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, feature):

        probs = self.model.predict_proba(feature)[0]
        
        return np.amax(probs), self.model.classes_[np.argmax(probs)]