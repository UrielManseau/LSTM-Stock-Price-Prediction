# LSTM-Stock-Price-Prediction

Utilisation d'un réseau de neurones récurrents (RNN), plus spéicifiquement l'architecture Long Short Term Memory (LSTM), afin de prédire le  prix de fermeture d'une action d'une compagnie cotée en bourse pour la journée suivante. Le programme utilise les données des 60 derniers jours sur le prix de fermeture du titre en question. L'architecture Long Short term momory a été privilégiée car elle est bien adapatée pour traiter des données sous forme de séries temporelles, les prix de fermetures d'une action sur une période de temps est un exemple de série temporelle. 

Ce programme arrive avec un résultat en procédant à une régression. L'algorithme qui performe la descente du gradient est Adam. Adam regroupe les avantages des algorithmes AdaGrad et RMSProp, ce qui explique pourquoi il est souvent séléctionné par défaut en intelligence artificelle. À noter que je ne fais pas de vérification si le modèle overtfit ou underfit. Si le modèle underfit ou overfit, on pourrait conclure que le modèle ne généralise pas bien et que s'il performe bien c'est simplement parce qu'il a appris ''par coeur'' le training set, ce qui est à éviter lorsqu'on construit un modèle d'apprentisage. Les résultats de ce programme ne devraient jamais être utilisés pour prendre des décisions financières. 

Les prochaines modifications à ce programme porteront sur le diagnostique du modèle pour vérifier s'il y a overfitting ou underfitting. Si ce diagnostique montre que le modèle overfit ou underfit, il sera nécessaire de prendre des mesures pour corriger la situation comme l'augmentation de la complexité du modèle (par ajout de couches) ou bien en utilisant la technique du early stopping (qui nous donnera un indicateur sur le nombre d'itérations qui peuvent être effectuées avant que le modèle commence à overfit). This program oversimplifies AI. 