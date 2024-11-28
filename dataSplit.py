import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('ratings.csv')


data = data[['userId', 'movieId', 'rating', 'timestamp']]
train, test = train_test_split(data, test_size=0.2, random_state=42)

train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)