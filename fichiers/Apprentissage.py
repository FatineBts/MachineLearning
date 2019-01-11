#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

from .Pretraitement import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression

############################ Etape 4 : Méthodes d'apprentissage ##################################

class Apprentissage: 
	def Naive_Bayes(data_train, data_test, target_train, target_test):
		#classifieur 
		classifier = GaussianNB()
		#apprentissage 
		clf = classifier.fit(data_train, target_train)
		#Exécution de la prédiction sur les données d'apprentissage
		target_pred = classifier.predict(data_test) #résultats obtenus
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : nous observons ici un mauvais entrainement. Les résultats obtenus en tests sont cohérents avec les valeurs d'entrainement.")
		# qualité de la prédiction
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))	

	def KNN(data_train, data_test, target_train, target_test):
		classifier = neighbors.KNeighborsClassifier(n_neighbors=1) 
		clf = classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : comme attendu les résultats obtenus lors de la phase d'entrainement sont bons.")
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))

	def perceptron_multi_couches(data_train, data_test, target_train, target_test): 
		classifier = MLPClassifier() 
		clf = classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : en observant la matrice de confusion on voit que le perceptron multi-couche a un entrainement qui n'est pas tout à fait exact. Le pourcentage obtenu est donc cohérent.")
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))

	def arbre_de_decision(data_train, data_test, target_train, target_test): 
		#pour entrainer et prendre des decisions
		classifier = DecisionTreeClassifier(criterion='entropy')
		clf = classifier.fit(data_train, target_train)
		#pour faire des prédictions 
		target_pred = classifier.predict(data_test)
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : comme attendu les résultats obtenus lors de la phase d'entrainement sont bons.")
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))

	def random_forest(data_train, data_test, target_train, target_test):
		classifier = RandomForestClassifier(n_estimators=100,criterion='entropy')
		clf = classifier.fit(data_train,target_train)
		target_pred = classifier.predict(data_test)
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : comme attendu les résultats obtenus lors de la phase d'entrainement sont bons.")
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))

	def regression_logistique(data_train, data_test, target_train, target_test):
		print("La régression logistique met plus de temps à s'achever. c'est pourquoi elle se trouve à la fin du code.")
		classifier= LogisticRegression(multi_class='auto')
		clf = classifier.fit(data_train,target_train)
		target_pred = classifier.predict(data_test) #résultats obtenus
		target_pred_train = classifier.predict(data_train)
		print("Matrice de confusion test :")
		Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Matrice de confusion train :")
		Pretraitement.matrice_de_confusion(target_train,target_pred_train)
		print("Remarque comparaison des matrices : nous observons ici entrainement moyen.")
		# qualité de la prédiction
		print("Qualité de la prédiction : :",np.mean(cross_val_score(clf,data_train,target_train, cv=5)))
