#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

############################## Etape 3 : pré-traitements et construction des descripteurs ##################################

class Pretraitement: 
	#pour séparer les données selon la méthode du cross-validation
	def separation_donnees(data):
		y = data.Cover_Type #donnees 
		X=data.drop('Cover_Type',axis=1) #classes 
		data_train, data_test, target_train, target_test = train_test_split(X,y, random_state=0, train_size=0.8) #80% apprentissage et 20% test 
		print("Séparation des données faite !")
		return data_train, data_test, target_train, target_test

	def matrice_de_confusion(target_test,target_pred):
		#matrice de confusion
		conf = confusion_matrix(target_test, target_pred)
		print("Matrice de confusion",conf)
		#pour visualiser les valeurs bien représentées et celles qui ne le sont pas 
		plt.matshow(conf, cmap='rainbow');
		plt.show()

	def epuration_donnees(data):
		#Etape 1 : équilibrage entre les classes 
		#à faire ...
		
		#Etape 2 : Une nouvelle mesure de la distance
		#fusion de Vertical_Distance_To_Hydrology et Horizontal_Distance_To_Hydrology : distance euclidienne 

		data['Horizontal_Distance_To_Hydrology'] = np.sqrt(np.square(data['Vertical_Distance_To_Hydrology']) + np.square(data['Horizontal_Distance_To_Hydrology']))
		data.rename({'Horizontal_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		data.drop(["Vertical_Distance_To_Hydrology"],axis = 1, inplace = True)		 

		#Etape 3 : Regroupement des variables Hillshade

		#fusion de Hillshade_9am et Hillshade_3pm : addition
		data['Hillshade_9am'] = np.abs(data['Hillshade_9am'] - data['Hillshade_3pm'])
		data.rename({'Hillshade_9am':'Hillshade_fusion'}, axis = 1, inplace = True)
		data.drop(["Hillshade_3pm"], axis = 1, inplace = True)

		#separation des données 
		y = data.Cover_Type
		X = data.drop('Cover_Type',axis=1)
		data_train, data_test, target_train, target_test = train_test_split(X,y, random_state=0, train_size=0.8)

		return data_train, data_test, target_train, target_test #on revoit data au final 
