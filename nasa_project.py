import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#capture of data: Cácere, city of Mato Grosso, Brasil
dados = pd.read_csv('dados_definitivo.csv') #The dataset contains informations about: Wind velocity, Temperature, Humidity, Rain fall, Longitude and Latitude, etc.
df1 = pd.DataFrame(dados)

#data training for LATITUDE
X = df1.drop('latitude', 1)
y = df1['latitude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

ln = LinearRegression()
ln.fit(X_train, y_train)
pred = ln.predict(X_test)
print("The next latitudes: ", pred)

#Mean squared error
RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("The error in Latitude is: ", RMSE)

#data training for LONGITUDE
X1 = df1.drop('longitude', 1)
y1 = df1['longitude']

#training test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)

ln1 = LinearRegression()
ln1.fit(X_train, y_train)

#prediction of next longitudes
pred1 = ln1.predict(X_test)
print("\nThe next longitude ar: ", pred1)

#Mean squared error
RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred1))
print(RMSE)

#Plot of graph 2d
plt.plot([y], [y1], 'bo')
plt.plot([pred], [pred1], 'ro')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Gráfico da Localização da Predição das Chamas')
