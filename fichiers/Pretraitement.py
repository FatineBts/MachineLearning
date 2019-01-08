#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

from .Manipulation_donnees import *
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate #cross validation

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
		print("Matrice de confusion\n",conf)
		#pour visualiser les valeurs bien représentées et celles qui ne le sont pas 
		plt.matshow(conf, cmap='rainbow');
		plt.show()

	def epuration_donnees(data):
		#Etape 1 : équilibrage entre les classes 
		#a)récupérer la classe avec le plus petit nombre d'éléments 
		
		classe = [0,0,0,0,0,0,0,0]

		for arbre in data.values: 
			if(int(arbre[54])==1): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[1]+=1
			elif(int(arbre[54])==2): #si l'élément vaut 2 alors on le met dans classe 2 
				classe[2]+=1
			elif(int(arbre[54])==3): #si l'élément vaut 3 alors on le met dans classe 3 
				classe[3]+=1
			elif(int(arbre[54])==4): #si l'élément vaut 4 alors on le met dans classe 4 
				classe[4]+=1
			elif(int(arbre[54])==5): #si l'élément vaut 5 alors on le met dans classe 5 
				classe[5]+=1
			elif(int(arbre[54])==6): #si l'élément vaut 6 alors on le met dans classe 6 
				classe[6]+=1
			elif(int(arbre[54])==7): #si l'élément vaut 7 alors on le met dans classe 7 
				classe[7]+=1
			else: 
				print("Pas de classe. Problème.")

		min_classe = min(classe[1],classe[2],classe[3],classe[4],classe[5],classe[6],classe[7])
		ligne = 2*min_classe
		nombre_l,_ = data.shape
		print("Nombre de données après traitement : ",nombre_l)

		print("classe 1",classe[1]) #nombre d'éléments dans chaque classe 
		print("classe 2",classe[2])
		print("classe 3",classe[3])
		print("classe 4",classe[4])
		print("classe 5",classe[5])
		print("classe 6",classe[6])
		print("classe 7",classe[7])

		nombre_l,_ = data.shape

		#b) trier selon la classe à laquelle la ligne appartient 
		data = data.sort_index(by='Cover_Type') #axis = 1 (colonnes)
		un = classe[1]-ligne
		#c) ne garder que 2 fois ce nombre pour chaque classe 
		data = data.drop(data.index[:un], axis=0) #classe 1 
		data = data.drop(data.index[5494:282000], axis=0) #classe 2 
		data = data.drop(data.index[15289:44289], axis=0) #classe 3 
		data = data.drop(data.index[26000:31000], axis=0) #classe 5
		data = data.drop(data.index[30000:42000], axis=0) #classe 6
		data = data.drop(data.index[38000:57000], axis=0) #classe 7

		classe = [0,0,0,0,0,0,0,0]
		
		for arbre in data.values: 
			if(int(arbre[54])==1): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[1]+=1
			elif(int(arbre[54])==2): #si l'élément vaut 2 alors on le met dans classe 2 
				classe[2]+=1
			elif(int(arbre[54])==3): #si l'élément vaut 3 alors on le met dans classe 3 
				classe[3]+=1
			elif(int(arbre[54])==4): #si l'élément vaut 4 alors on le met dans classe 4 
				classe[4]+=1
			elif(int(arbre[54])==5): #si l'élément vaut 5 alors on le met dans classe 5 
				classe[5]+=1
			elif(int(arbre[54])==6): #si l'élément vaut 6 alors on le met dans classe 6 
				classe[6]+=1
			elif(int(arbre[54])==7): #si l'élément vaut 7 alors on le met dans classe 7 
				classe[7]+=1
			else: 
				print("Pas de classe. Problème.")

		nombre_lignes,_ = data.shape
		print("Nombre de données après traitement : ",nombre_lignes)

		print("classe 1",classe[1]) #nombre d'éléments dans chaque classe 
		print("classe 2",classe[2])
		print("classe 3",classe[3])
		print("classe 4",classe[4])
		print("classe 5",classe[5])
		print("classe 6",classe[6])
		print("classe 7",classe[7])

		somme_proba = 0
		proba = np.zeros(7)
		proba[0] = classe[1]/nombre_lignes
		proba[1] = classe[2]/nombre_lignes
		proba[2] = classe[3]/nombre_lignes
		proba[3] = classe[4]/nombre_lignes
		proba[4] = classe[5]/nombre_lignes
		proba[5] = classe[6]/nombre_lignes
		proba[6] = classe[7]/nombre_lignes
		somme_proba = np.sum(proba)
		print("\nExample : nombre d'arbres appartenant à la classe 1 : ",classe[1],"\nExample : probabilité d'appartenir à la classe 1 : ",proba[3],"\nProba équivalente : 1/7. Somme des probas = 1 ? : ",somme_proba)
		
		#d) completer pour avoir plus de données 
		#A faire

		#Etape 2 : Une nouvelle mesure de la distance
		#fusion de Vertical_Distance_To_Hydrology et Horizontal_Distance_To_Hydrology : distance euclidienne 

		data['Horizontal_Distance_To_Hydrology'] = np.sqrt(np.square(data['Vertical_Distance_To_Hydrology']) + np.square(data['Horizontal_Distance_To_Hydrology']))
		data.rename({'Horizontal_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		data.drop('Vertical_Distance_To_Hydrology', axis = 1, inplace = True)		 

		#Etape 3 : Regroupement des variables Hillshade
		#fusion de Hillshade_9am et Hillshade_3pm : addition
		data['Hillshade_9am'] = np.abs(data['Hillshade_9am'] - data['Hillshade_3pm'])
		data.rename({'Hillshade_9am':'Hillshade_fusion'}, axis = 1, inplace = True)
		data.drop('Hillshade_3pm', axis = 1, inplace = True) #quand inplace = True, le data original est modifié directement

		#separation des données 
		y = data.Cover_Type
		X = data.drop('Cover_Type',axis=1)

		data_train, data_test, target_train, target_test = train_test_split(X,y, random_state=0, train_size=0.8)
		return data_train, data_test, target_train, target_test #on revoit data au final 
