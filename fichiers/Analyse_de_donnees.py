#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

from .Annexe import *
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# classe pour standardisation (ACP)
from sklearn.preprocessing import StandardScaler
# classe pour l'ACP
from sklearn.decomposition import PCA

############################## Etape 2 : analyse des données ##################################

class Analyse_de_donnees: 
	#première analyse des données, utilisation de Pandas
	def analyse_usuelle(data,element):
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
			print("Analyse Horizontal_Distance_To_Hydrology : On identifie une médiane égale environ à 200 et de nombreuses valeurs dépassant du boxplot.")
		if(type_element=="Vertical_Distance_To_Hydrology"): 
			print("Analyse Vertical_Distance_To_Hydrology : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
		if(type_element=="Horizontal_Distance_To_Roadways"): 
			print("Analyse Horizontal_Distance_To_Roadways : On identifie une médiane égale environ à 1200 et de nombreuses valeurs dépassant du boxplot.")
		if(type_element=="Hillshade_9am"): 
			print("Analyse Hillshade_9am : On identifie une médiane égale environ à 200 et quelques valeurs dépassant du boxplot.")
		if(type_element=="Hillshade_Noon"): 
			print("Analyse Hillshade_Noon : On identifie une médiane égale environ à 220 et de nombreuses valeurs dépassant du boxplot.")
		if(type_element=="Hillshade_3pm"): 
			print("Analyse Hillshade_3pm : On identifie une médiane égale environ à 150 et de nombreuses valeurs dépassant du boxplot.")
		if(type_element=="Horizontal_Distance_To_Fire_Points"): 
			print("Analyse Horizontal_Distance_To_Fire_Points : On identifie une médiane égale environ à 1800 et de nombreuses valeurs dépassant du boxplot.")

		if(show==True):
			print("Analyse complète : ") 
			print("Analyse Elevation : On identifie une médiane égale environ à 3000 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Aspect : On remarque que 50 pour cent des arbres ont une orientation inférieure à 125 degrés. De plus l'écart inter-quartile est important ce qui traduit beaucoup de données. Etant donné que les moustaches sont longues, on comprend également que les valeurs sont étendues.")
			print("Analyse Slope : On voit que la médiane vaut environ 12 et que beaucoup de valeurs sont mal placées, il faudra donc épurer cette variable par la suite.")
			print("Analyse Horizontal_Distance_To_Hydrology : On identifie une médiane égale environ à 200 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Vertical_Distance_To_Hydrology : On identifie une médiane égale environ à 20 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Horizontal_Distance_To_Roadways : On identifie une médiane égale environ à 1200 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Hillshade_9am : On identifie une médiane égale environ à 200 et quelques valeurs dépassant du boxplot.")
			print("Analyse Hillshade_Noon : On identifie une médiane égale environ à 220 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Hillshade_3pm : On identifie une médiane égale environ à 150 et de nombreuses valeurs dépassant du boxplot.")
			print("Analyse Horizontal_Distance_To_Fire_Points : On identifie une médiane égale environ à 1800 et de nombreuses valeurs dépassant du boxplot.")

		plt.show()

	#pour l'analyse en composantes principales 
	def ACP(data):
		# Nous devons explicitement centrer et réduire les données pour réaliser une ACP normée avec PCA
		# Instanciation
		sc = StandardScaler()
		Z = sc.fit_transform(data)
		# On ne garde que les variables quantitatives
		Z = Z[:,0:10]
		# On peut vérifier les propriétés de ces nouvelles données
		# Moyenne
		print ("Le tableau des moyennes est : ", np.mean(Z,axis=0))
		print ("Aux erreurs de troncature près, les moyennes valent 0")
		# Ecart-type
		print ("Le tableau des ecart-type est : ", np.std(Z,axis=0,ddof=0))

		# Nous pouvons maintenant faire l'ACP
		# Instanciation
		acp = PCA(svd_solver='full')
		# Affichage des paramètres
		print(acp)

		#calculs
		coord = acp.fit_transform(Z)

		# On peut utiliser la "cassure" pour déterminer le nombre de facteurs à retenir
		p = acp.n_components_
		print ("Avec les 3 premières variables, on a 70 pourcents de l'information")
		print ("Avec les 4 premières variables, on a 80 pourcents de l'information")
		print ("Avec les 6 premières variables, on a 90 pourcents de l'information")

		plt.plot(np.arange(1,p+1),np.cumsum(acp.explained_variance_ratio_))
		plt.title("Explained variance vs. # of factors")
		plt.ylabel("Cumsum explained variance ratio")
		plt.xlabel("Factor number")
		plt.show()
		n = Z.shape[0]
		eigval = acp.singular_values_**2/n

		# 2e méthode : Test des bâtons brisés
		#seuils pour test des bâtons brisés
		print("Méthode 2 : Test des bâtons brisés :")
		n = Z.shape[0]
		bs = np.zeros(p)
		compteur = 0
		for i in range(p):
			bs[i] = 1.0/float(p-i)
		bs = np.cumsum(bs)
		bs = bs[::-1]
		print(pandas.DataFrame({'Val.Propre':eigval,'Seuils':bs}))

		# Qualité de représentation
		#contribution des individus dans l'inertie totale
		di = np.sum(Z**2,axis=1)
		print(pandas.DataFrame({'ID':range(n),'d_i':di}))
		#qualité de représentation des individus 
		cos2 = coord**2
		for j in range(p):
			cos2[:,j] = cos2[:,j]/di
		print(pandas.DataFrame({'id':range(n),'COS2_1':cos2[:,0],'COS2_2':cos2[:,1],'COS2_3':cos2[:,2]}))
		#vérifions la théorie 
		print(np.sum(cos2,axis=1))
		print ("On a bien la somme des cos égale à 1.\n")

		#max des cos
		max_cos2 = [0,0,0]
		mean_cos2 = [0,0,0]
		for i in range(0,3):
			mean_cos2[i] = np.mean(cos2[:,i])
			max_cos2[i] = max(cos2[:,i])
		print("Les contributions les plus importantes sont : ",max_cos2[0], max_cos2[1], max_cos2[2])

		nombre_lignes = Annexe.nombre_de_lignes(data)

		for i in range(0,nombre_lignes): 
			for j in range(0,3):
				if(cos2[i,j]==max_cos2[j]): 
					print("Axe numéro",j,"Arbre numéro :",i) #pour avoir la variable (arbre) correspondant au max_ctr

		#probabilité d'éléments bien positionnés 
		proba = [0,0,0]
		for i in range(0,nombre_lignes): 
			for j in range(0,3):
				if(cos2[i,j]>mean_cos2[j]):
					proba[j]+=1
		
		for j in range(0,3):
			proba[j]/=nombre_lignes
			print("Axe numéro",j,"Proba :",proba[j]) #pour avoir la variable (arbre) correspondant au max_ctr
		print("Les moyennes sont (axe1,axe2,axe3) : ",mean_cos2[0], mean_cos2[1], mean_cos2[2])

		#contributions aux axes
		ctr = coord**2
		for j in range(p):
			ctr[:,j] = ctr[:,j]/(n*eigval[j])
		print(pandas.DataFrame({'id':range(n),'CTR_1':ctr[:,0],'CTR_2':ctr[:,1],'CTR_3':ctr[:,2]}))

		max_ctr = [0,0,0]
		mean_ctr = [0,0,0]
		for i in range(0,3):
			max_ctr[i] = max(ctr[:,i])
			mean_ctr[i] = np.mean(ctr[:,i])

		print("Les contributions les plus importantes sont : ",max_ctr[0], max_ctr[1], max_ctr[2])
		print("Les moyennes des contributions sont : ",mean_ctr[0], mean_ctr[1], mean_ctr[2])

		for i in range(0,nombre_lignes): 
			for j in range(0,3):
				if(ctr[i,j]==max_ctr[j]): 
					print("Axe numéro",j,"Arbre numéro :",i) #pour avoir la variable (arbre) correspondant au max_ctr

		#racine carrée des valeurs propres
		sqrt_eigval = np.sqrt(eigval)
		#corrélation des variables avec les axes
		corvar = np.zeros((p,p))
		for k in range(p):
			corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
		#cercle des corrélations
		fig, axes = plt.subplots(figsize=(8,8))
		axes.set_xlim(-1,1)
		axes.set_ylim(-1,1)
		#affichage des étiquettes(noms des variables)
		index = ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"]
		for j in range(p):
			plt.annotate(index[j],(corvar[j,0],corvar[j,1]))
		#ajouter les axes
		plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
		plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
		#ajouter un cercle
		cercle = plt.Circle((0,0),1,color='blue',fill=False)
		axes.add_artist(cercle)
		#affichage
		plt.show()

	def correlation_variables(data): 
		X=data[:,:10]
		y=data[:,54] 
		df = pandas.DataFrame(X)
		df.head()
		print("Corrélation :", df.corr())