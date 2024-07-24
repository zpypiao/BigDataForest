import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from data import load_data
import datetime
import pickle

def save_model(model):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'./model/random_forest/{current_time}.pth'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)



X, y = load_data('./数据/cno.csv')

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print('rmse {}'.format(rmse))
plt.plot(y_test[:100], label='y_test')
plt.plot(preds[:100], label='preds')
plt.legend()
plt.show()

save_model(model)