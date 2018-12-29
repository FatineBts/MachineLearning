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
	proba =np.zeros(7)
	proba[0] = classe[1]/nombre_lignes
	proba[1] = classe[2]/nombre_lignes
	proba[2] = classe[3]/nombre_lignes
	proba[3] = classe[4]/nombre_lignes
	proba[4] = classe[5]/nombre_lignes
	proba[5] = classe[6]/nombre_lignes
	proba[6] = classe[7]/nombre_lignes
	somme_proba = np.sum(proba)

	plt.bar(range(7),proba)
	plt.xlabel('Type de couverture')
	plt.ylabel('Probabilité')
	plt.show()

	print("Nombre de lignes : ",nombre_lignes, "\nExample : nombre d'arbres appartenant à la classe 1 : ",classe[1],"\nExample : probabilité d'appartenir à la classe 1 : ",proba[0],"\nSomme des probas = 1 ? : ",somme_proba)

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

#croisement de variables et étude de l'influence de l'une sur l'autre 
def croisement_de_variables(data):
	print("Etude de l'impact de la présence (1) ou absence (0) du sol de type 1 (Cathedral family) sur la présence (1) ou l'absence (0) de zone sauvage de type Rawah : ")
	print(pandas.crosstab(data['Wilderness_Area'],data['Soil_Type'],normalize='index')) #utilisée pour des variables qualitatives 
	print("Analyse : On remarque lorsqu'il y a une zone sauvage, la probabilité de présence d'un sol de type 1 est nulle. On remarque également que lorsqu'il y a ou pas un sol de type 1, nous avons presque autant de chance de retrouver une zone sauvage que de ne pas en trouver. On peut donc conclure que la présence de zone sauvage semble influencer la présence d'un sol de type 1 tandis que l'inverse est faux.\n")
 
	print("Etude de l'impact de la présence (1) ou absence (0) du sol de type 2 (Vanet) sur la présence (1) ou l'absence (0) de zone sauvage de type Neota : ")
	print(pandas.crosstab(data['Wilderness_Area.1'],data['Soil_Type.1'],normalize='index'))
	print("Analyse : Même analyse.")

#histogramme
def histo(data,type_element,show):
	data.hist(column=type_element) #abscisses = Elevation, ordonnée = quantitée 	
	if(type_element=="Elevation"):
		print("Analyse : On peut voir que beaucoup d'arbres ont une élévation comprise entre 2850 et 3250 m environ.")
	if(type_element=="Aspect"): #faut comprendre ce que c'est ... pas trop compris je t'avoue 
		print("Analyse : On observe un aspect un peu hétérogène avec des valeurs plus importantes entre 0 et 150 ou encore 300 et 500 degrés Azimut.")
	if(type_element=="Slope"): 
		print("Analyse : On constate que peu d'arbres sont fortement inclinés et que la plupart des arbres ont une inclinaison de 10 degrés.")
	if(type_element=="Horizontal_Distance_To_Hydrology"): 
		print("Analyse : On remarque que la plupart des arbres ont une distance horizontale proche d'un point d'eau.")
	if(type_element=="Vertical_Distance_To_Hydrology"): 
		print("Analyse : On remarque que la plupart des arbres se trouvent à une distance verticale équivalente à 100 m d'un point d'eau.")
	if(type_element=="Horizontal_Distance_To_Roadways"): 
		print("Analyse : On peut voir que la distance horizontale à la route la plus paroche diverge fortement.")
	if(type_element=="Hillshade_9am"): 
		print("Analyse : On remarque que la valeur de l'ombrage à 9h est élevée pour la plupart des arbres.")
	if(type_element=="Hillshade_Noon"): 
		print("Analyse : On remarque que la valeur de l'ombrage à midi est élevée pour la plupart des arbres.")
	if(type_element=="Hillshade_3pm"): 
		print("Analyse : On remarque que la valeur de l'ombrage à 15h est élevée pour certains arbres et vaut entre 100 et 200.")
	if(type_element=="Horizontal_Distance_To_Fire_Points"): 
		print("Analyse : On remarque que la plupart des arbres ont une distance au départ du feu le plus proche comprise entre 800 et 2800 m.")

	if(show==True):
		print("Analyse complète : ") 
		print("On peut voir que beaucoup d'arbres ont une élévation comprise entre 2850 et 3250 m environ.")
		print("On observe un aspect un peu hétérogène avec des valeurs plus importantes entre 0 et 150 ou encore 300 et 500 degrés Azimut.")
		print("On constate que peu d'arbres sont fortement inclinés et que la plupart des arbres ont une inclinaison de 10 degrés.")
		print("On remarque que la plupart des arbres ont une distance horizontale proche d'un point d'eau.")
		print("On remarque que la plupart des arbres se trouvent à une distance verticale équivalente à 100 m d'un point d'eau.")
		print("On peut voir que la distance horizontale à la route la plus paroche diverge fortement.")
		print("On remarque que la valeur de l'ombrage à 9h est élevée pour la plupart des arbres.")
		print("On remarque que la valeur de l'ombrage à midi est élevée pour la plupart des arbres.")
		print("On remarque que la valeur de l'ombrage à 15h est élevée pour certains arbres et vaut entre 100 et 200.")
		print("On remarque que la plupart des arbres ont une distance au départ du feu le plus proche comprise entre 800 et 2800 m.")

	plt.show()

#boxplot 
def boxplot(data,type_element,show):
	data.boxplot(column=type_element)

	if(type_element=="Elevation"):
		print("Analyse Elevation : On identifie une médiane égale environ à 3000 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Aspect"): #faut comprendre ce que c'est ... pas trop compris je t'avoue 
		print("Analyse Aspect : On remarque que 50 pour cent des arbres ont une orientation inférieure à 125 degrés. De plus l'écart inter-quartile est important ce qui traduit beaucoup de données. Etant donné que les moustaches sont longues, on comprend également que les valeurs sont étendues.")
	if(type_element=="Slope"): 
		print("Analyse Slope : On voit que la médiane vaut environ 12 et que beaucoup de valeurs sont mal placées, il faudra donc épurer cette variable par la suite.")
	if(type_element=="Horizontal_Distance_To_Hydrology"): 
		print("Analyse Horizontal_Distance_To_Roadways : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Vertical_Distance_To_Hydrology"): 
		print("Analyse : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Horizontal_Distance_To_Roadways"): 
		print("Analyse Horizontal_Distance_To_Roadways : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Hillshade_9am"): 
		print("Analyse Hillshade_9am : On identifie une médiane égale environ à 2000 et quelques valeurs dépassant du boxplot.")
	if(type_element=="Hillshade_Noon"): 
		print("Analyse Hillshade_Noon : On identifie une médiane égale environ à 220 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Hillshade_3pm"): 
		print("Analyse Hillshade_3pm : On identifie une médiane égale environ à 180 et de nombreuses valeurs dépassant du boxplot.")
	if(type_element=="Horizontal_Distance_To_Fire_Points"): 
		print("Analyse Horizontal_Distance_To_Fire_Points : On identifie une médiane égale environ à 1800 et de nombreuses valeurs dépassant du boxplot.")

	if(show==True):
		print("Analyse complète : ") 
		print("Analyse Elevation : On identifie une médiane égale environ à 3000 et de nombreuses valeurs dépassant du boxplot.")
		print("Analyse Aspect : On remarque que 50 pour cent des arbres ont une orientation inférieure à 125 degrés. De plus l'écart inter-quartile est important ce qui traduit beaucoup de données. Etant donné que les moustaches sont longues, on comprend également que les valeurs sont étendues.")
		print("Analyse Slope : On voit que la médiane vaut environ 12 et que beaucoup de valeurs sont mal placées, il faudra donc épurer cette variable par la suite.")
		print("Analyse Horizontal_Distance_To_Hydrology : On identifie une médiane égale environ à 200 et de nombreuses valeurs dépassant du boxplot.")
		print("Analyse Horizontal_Distance_To_Roadways : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
		print("Analyse Hillshade_9am : On identifie une médiane égale environ à 2000 et quelques valeurs dépassant du boxplot.")
		print("Analyse Hillshade_Noon : On identifie une médiane égale environ à 220 et de nombreuses valeurs dépassant du boxplot.")
		print("Analyse Hillshade_3pm : On identifie une médiane égale environ à 180 et de nombreuses valeurs dépassant du boxplot.")
		print("Analyse Horizontal_Distance_To_Fire_Points : On identifie une médiane égale environ à 1800 et de nombreuses valeurs dépassant du boxplot.")

	plt.show()



###################### Appel de fonctions #########################

fichier = "covtype.data"
fichier_modifie = "covtype_modifie.data"

affichage("lecture_fichier")
data = lecture_fichier(fichier)
print("\n")

affichage("moyenne")
moyenne(data)
print("\n")

affichage("type_foret") 
type_foret(data)
print("\n")

affichage("lecture_fichier_Numpy") 
data_np = lecture_fichier_Numpy(fichier)
print("\n")

affichage("lecture_fichier_Pandas") 
data_pandas = lecture_fichier_Pandas(fichier_modifie)
print("\n")

affichage("analyse_basique") 
element = 0 #0 = Elevation, 1 = Aspect, 2 = Slope ....
analyse_basique(data_pandas,element)
print("\n")

affichage("croisement_de_variables") 
croisement_de_variables(data_pandas)
print("\n")

affichage("histogramme") 
type_element = 'Horizontal_Distance_To_Fire_Points'
show = True
histo(data_pandas,type_element,show)
print("\n")

affichage("boxplot") 
type_element='Elevation'
show = True 
boxplot(data_pandas,type_element,show)
