from sklearn.datasets import load_boston
import pandas as pd


# load data
boston_dataset = load_boston()
data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
data['MEDV'] = boston_dataset.target

# create train_set and test_set once.
# data = data.sample(frac=1).reset_index(drop=True)
# train_set = data.head(int(0.8 * 506))
# test_set = data.tail(506 - int(0.8 * 506))
# train_set.to_csv("./train_set.csv", index=False)
# test_set.to_csv("./test_set.csv", index=False)

# load train_set and test_set
# train_set = pd.read_csv('./train_set.csv')
# test_set = pd.read_csv('./test_set.csv')
