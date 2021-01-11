# Utilisation d'un réseau de neurones récurrents, plus spéicifiquement le réseau Long Short Term Memory (LSTM), afin de prédire le 
# prix de fermeture d'une action d'une compagnie cotée en bourse. Les données des 60 derniers jours seront utilisées. 

#Importation des librairies 
import pandas_datareader as web
import numpy as np 
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('fivethirtyeight')

#Obtenir le data sur le stock
symbole = input("Entrez le symbole du titre : ")
debut = datetime(int(input("Année IPO : ")), int(input("Mois IPO : ")), int(input("Jour IPO :")))
fin = datetime.today().strftime("%Y-%m-%d")
data = web.DataReader(symbole, data_source = 'yahoo', start = debut, end = fin)

#Obtenir le nombre de colonnes et de rangées du data
data.shape

#Visualisation de l'historique du prix de fermeture
plt.figure(figsize=(16,8))
plt.title("Historique du prix de fermeture de " + symbole)
plt.plot(data['Close'])
plt.ylabel('Prix de fermeture USD ($)', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.show()

#Créer un nouveau tableau de données avec seulement le prix de fermeture
dataFermeture = data.filter(['Close'])
dataset = dataFermeture.values

trainingDataLen = math.ceil(len(dataset)*0.8)
trainingDataLen

scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataset)
trainData = scaledData[0:trainingDataLen, :]

#Séparer le data en xTrain et yTrain
xTrain = []
yTrain = []
for i in range(60, len(trainData)):
    xTrain.append(trainData[i-60:i,0])
    yTrain.append(trainData[i,0])
    if i <= 60:
        print(xTrain)
        print(yTrain)
        print()
           
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
 
#Transformer xTrain pour que ce soit un vecteur tridimensionel  
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1],1))

#Construire modèle Long short-term memory
#2 couches LSTM de 50 neurones chaque 
#2 couches denses (regulières), une de 25 neurones et l'autre de 1 neurones
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (xTrain.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compiler le modèle
model.compile(optimizer='adam', loss ='mean_squared_error')

#Entraîner le modèle
model.fit(xTrain, yTrain, batch_size=1,epochs=1)

#Le traning data aura des valeurs scaled de 0 à 1 et représentera 20% des données
testData = scaledData[trainingDataLen - 60:, :]

#Créer les data sets de test xTest et yTest
xTest = []
yTest = dataset[trainingDataLen:, :]

for i in range(60, len(testData)):
    xTest.append(testData[i-60:i, 0])
    
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTrain.shape[1], 1))

#Obtenir les prix prédits par le modèle
preds = model.predict(xTest)
preds = scaler.inverse_transform(preds)

#Obtenir le RMSE (root mean squared error) du modèle 
rmse = np.sqrt(np.mean(((preds - yTest)**2))) 

#Visualisation des prédictions du modèle et de la réalité dans un graphique
entraînement = data[:trainingDataLen]
réalité = data[trainingDataLen:]
pd.set_option('mode.chained_assignment', None)
réalité['preds'] = preds
plt.figure(figsize =(16,8))
plt.title('Modèle prédictif')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Prix de fermeture USD ($)', fontsize =18)
plt.plot(entraînement['Close'])
plt.plot(réalité[['Close', 'preds']])
plt.legend(['Entraînement', 'Réalité', 'prédictions'], loc = 'lower left')
plt.show()

new_quote = web.DataReader(symbole, data_source='yahoo', start= debut, end=fin)

#Créer un nouveau data frame
df = new_quote.filter(['Close'])

#Obtenir les 60 derniers prix de fermeture et convertir le data frame en un tableau
derniers60Jours = df[-60:].values
#Scaler le data
derniers60JoursScaled = scaler.transform(derniers60Jours)

#Créer un tableau pour les tests
xTest = []
xTest.append(derniers60JoursScaled)
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1],1 ))
#Obtenir les valeurs prédites
prixPred = model.predict(xTest)

#De-scale le data et présnter les résultats
prixPred = scaler.inverse_transform(prixPred)
print('Le prix de fermeture prédit pour demain du symbole ' + symbole + ' est :')
print (prixPred)
















