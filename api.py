import numpy as np
import pickle

pipeline = pickle.load(open('./model/model.pkl', 'rb'))

example = {
  'Pclass': 3,  # int
  'Sex': 'M',    # M or F
  'Age': 22,    # int
  'SibSp': 1,  # int
  'Parch': 0,  # int
  'Fare': 7.25    # float
}

def make_prediction(features):
    X = np.array([features['Pclass'], int(features['Sex'] == 'M'), features['Age'],
                  features['SibSp'], features['Parch'], features['Fare']]).reshape(1,-1)
    prob_survived = pipeline.predict_proba(X)[0, 1]

    result = {
        'prediction': int(prob_survived > 0.5),
        'prob_survived': prob_survived
    }
    return result

if __name__ == '__main__':
    print(make_prediction(example))
