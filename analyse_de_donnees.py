#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
import matplotlib.pyplot as plt

#lecture du fichier 
def lecture_fichier(fichier): 
	lignes = open(fichier, "r")
	data = lignes.readlines() #va permettre de lire le fichier 
	return data

def lecture_fichier_Numpy(fichier): 
	data = np.loadtxt("covtype.data",delimiter=",") #autre type d'utilisation des données 
	return data

def lecture_fichier_Pandas(fichier):
	data = pandas.read_table("covtype_modifie.data",sep = ',',header = 0)
	return data

def nombre_de_lignes(data): 
	nombre_lignes = 0
	for lignes in data: #pour avoir le nombre de lignes 
		nombre_lignes+=1
	return nombre_lignes

def affichage(nom):
	print("################################ Fonction",nom," : #########################################")

#calcule les valeurs moyennes pour les données quantitatives uniquement
def moyenne(data):

	nombre_lignes=nombre_de_lignes(data)
	somme = 0
	moyenne= [0,0,0,0,0,0,0,0,0,0]

	for lignes in data:
		for i in range(0,10): #pour les variables quantitatives
			moyenne[i]+=int(lignes.split(",")[i])
	
	for i in range(0,10): 
		moyenne[i]/=nombre_lignes #pour diviser par le nombre de lignes 
	
	print("La moyenne des éléments quantitatifs obtenue à la main est : ",moyenne)

#donne la probabilité d'appartenance à chaque classe de forêt 
def type_foret(data): 
	classe = [0,0,0,0,0,0,0,0]
	nombre_lignes = nombre_de_lignes(data)
	proba1 = 0
	proba2 = 0
	proba3 = 0
	proba4 = 0
	proba5 = 0
	proba6 = 0
	proba7 = 0
	somme_proba = 0

	for arbre in data: 
			
		if(int(arbre.split(",")[54])==1): #si l'élément vaut 1 alors on le met dans classe 1 
			#print("ok")
			classe[1]+=1

		elif(int(arbre.split(",")[54])==2): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[2]+=1

		elif(int(arbre.split(",")[54])==3): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[3]+=1

		elif(int(arbre.split(",")[54])==4): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[4]+=1

		elif(int(arbre.split(",")[54])==5): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[5]+=1

		elif(int(arbre.split(",")[54])==6): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[6]+=1

		elif(int(arbre.split(",")[54])==7): #si l'élément vaut 1 alors on le met dans classe 1 
			classe[7]+=1
		
		else: 
			print("Pas de classe. Problème.")

	proba1 = classe[1]/nombre_lignes
	proba2 = classe[2]/nombre_lignes
	proba3 = classe[3]/nombre_lignes
	proba4 = classe[4]/nombre_lignes
	proba5 = classe[5]/nombre_lignes
	proba6 = classe[6]/nombre_lignes
	proba7 = classe[7]/nombre_lignes
	somme_proba = proba1 + proba2 + proba3 + proba4 + proba5 + proba6 + proba7

	print("Nombre de lignes : ",nombre_lignes, "\nExample : nombre d'arbres appartenant à la classe une : ",classe[1],"\nExample : probabilité d'appartenir à la classe une : ",proba1,"\nSomme des probas = 1 ? : ",somme_proba)

#première analyse des données, utilisation de Pandas
def analyse_basique(data,element):
	col = []
	nombre_lignes,colonne = data.shape #donne le nombre de lignes et de colonnes 

	#for i in range(0,nombre_lignes):
	#	col.append(data[i][element]) #remplit le vecteur colonne avec tous les éléments d'une colonne

	#affiche les premières lignes d'un jeu de données 
	print("Premières lignes du jeu de données : ")
	print(data.head())
	
	print("Description des données :")
	print(data.describe(include="all"))
	#plt.boxplot(data)
	#plt.show()
	print("Enumération des colonnes : ")
	print(data.columns)

	print("Accès à la colonne Elevation  : ")
	print(data['Elevation'].head()) #affichage que des premières valeurs 
###################### Appel de fonctions #########################

fichier = "covtype.data"
fichier_modifie = "covtype_modifie.data"

affichage("lecture_fichier")
data = lecture_fichier(fichier)

affichage("moyenne")
moyenne(data)

affichage("type_foret") 
type_foret(data)

affichage("lecture_fichier_Numpy") 
data_np = lecture_fichier_Numpy(fichier)

affichage("lecture_fichier_Pandas") 
data_pandas = lecture_fichier_Pandas(fichier_modifie)

affichage("analyse_basique") 
element = 0 #0 = Elevation, 1 = Aspect, 2 = Slope ....
analyse_basique(data_pandas,element)



