#!/usr/bin/env python
# -*- coding: utf-8 -*-

#pour la cross validation
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


############################ Etape 4 : Méthodes d'apprentissage ##################################

class Apprentissage: 
	def Naive_Bayes(data_train, data_test, target_train, target_test):
		print("Méthode de Gauss : Classification Naive Bayes, suppose que chaque classe est contruite à partir d'une distribution Gaussienne alignée. Avantage : très rapide.")
		#classifieur 
		classifier = GaussianNB()
		#apprentissage 
		classifier.fit(data_train, target_train)
		#Exécution de la prédiction sur les données d'apprentissage
		target_pred = classifier.predict(data_test) #résultats obtenus
		# qualité de la prédiction
		print("Qualité de la prédiction : ",accuracy_score(target_pred, target_test))
	
	def KNN(data_train, data_test, target_train, target_test):
		classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
		classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def perceptron_multi_couches(data_train, data_test, target_train, target_test): 
		classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001) 
		classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def arbre_de_decision(data_train, data_test, target_train, target_test): 
		#pour entrainer et prendre des decisions
		classifier = DecisionTreeClassifier()
		classifier.fit(data_train, target_train)
		#pour faire des prédictions 
		target_pred = classifier.predict(data_test)
		conf = confusion_matrix(target_test, target_pred)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def random_forest(data_train, data_test, target_train, target_test):
		classifier = RandomForestClassifier(n_estimators=30,criterion='entropy',max_features=None)
		classifier.fit(data_train,target_train)
		target_pred = classifier.predict(data_test)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))
