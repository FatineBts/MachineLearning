#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
# classe pour standardisation (ACP)
from sklearn.preprocessing import StandardScaler
# classe pour l'ACP
from sklearn.decomposition import PCA
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

############################## Etape 1 : manipulation des données ##################################

class Lecture: 
	#lecture du fichier 
	def lecture_fichier(fichier): 
		lignes = open(fichier, "r")
		data = lignes.readlines() #va permettre de lire le fichier 
		return data

	def lecture_fichier_Numpy(fichier): 
		data = np.loadtxt(fichier,delimiter=",") #autre type d'utilisation des données 
		return data

	def lecture_fichier_Pandas(fichier):
		data = pandas.read_csv(fichier,delimiter=",",header = 0)
		return data

class Annexe: 
	def nombre_de_lignes(data): 
		nombre_de_lignes = 0
		nombre_lignes,colonne = data.shape
		return nombre_lignes

	def affichage(nom):
		print("################################ Fonction",nom," : #########################################")

class Manipulation_donnees:
	#calcule les valeurs moyennes pour les données quantitatives uniquement
	def moyenne(data):

		nombre_lignes=Annexe.nombre_de_lignes(data)
		somme = 0
		moy = [0,0,0,0,0,0,0,0,0,0]

		for lignes in data:
			for i in range(0,10): #pour les variables quantitatives
				moy[i]+=int(lignes[i])
	
		for i in range(0,10): 
			moy[i]/=nombre_lignes #pour diviser par le nombre de lignes 
	
		print("La moyenne des éléments quantitatifs obtenue à la main est : ",moy)

	#donne la probabilité d'appartenance à chaque classe de forêt 
	def type_foret(data): 
		classe = [0,0,0,0,0,0,0,0]
		nombre_lignes = Annexe.nombre_de_lignes(data)
		somme_proba = 0

		for arbre in data: 
			if(int(arbre[54])==1): #si l'élément vaut 1 alors on le met dans classe 1 
				#print("ok")
				classe[1]+=1
			elif(int(arbre[54])==2): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[2]+=1
			elif(int(arbre[54])==3): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[3]+=1
			elif(int(arbre[54])==4): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[4]+=1
			elif(int(arbre[54])==5): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[5]+=1
			elif(int(arbre[54])==6): #si l'élément vaut 1 alors on le met dans classe 1 
				classe[6]+=1
			elif(int(arbre[54])==7): #si l'élément vaut 1 alors on le met dans classe 1 
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

############################## Etape 3 : pré-traitements et construction des descripteurs ##################################

class Pretraitement: 
	#pour séparer les données selon la méthode du cross-validation
	def separation_donnees(data):
		X=data[:,:53] # les données : toutes les lignes mais pas la dernière colonne des labels 
		y=data[:,54] # les labels que l'on converve : les classes 
		#print("X : ",X)
		#print("y :",y)
		#print("\n")
		data_x = train_test_split(X,y, random_state=0, train_size=0.8) #80% apprentissage et 20% test 
		print("Séparation des données selon le cross_validation fait !")
		#print("data_train, data_test, target_train, target_test : ",data_test)
		return data_x

	def matrice_de_confusion(target_test,target_pred):
		#matrice de confusion
		conf = confusion_matrix(target_test, target_pred)
		print("Matrice de confusion",conf)
		#pour visualiser les valeurs bien représentées et celles qui ne le sont pas 
		plt.matshow(conf, cmap='rainbow');
		plt.show()

	def epuration_donnees(data):
		#prémices 
		y = data.Soil_Type
		X=data.drop('Soil_Type',axis=1)
		data_train, data_test, target_train, target_test = train_test_split(X,y, random_state=0, train_size=0.8)

		#Etape 1 : équilibrage entre les classes 
		#à faire ...
		
		#Etape 2 : Une nouvelle mesure de la distance
		#fusion de Vertical_Distance_To_Hydrology et Horizontal_Distance_To_Hydrology : distance euclidienne 
		
		data_train['Vertical_Distance_To_Hydrology'] = np.sqrt(np.square(data_train['Vertical_Distance_To_Hydrology'] + np.square(data_train['Horizontal_Distance_To_Hydrology'])))
		data_test['Vertical_Distance_To_Hydrology'] = np.sqrt(np.square(data_test['Vertical_Distance_To_Hydrology'] + np.square(data_test['Horizontal_Distance_To_Hydrology'])))
		#on renomme la fusion des deux en VH_Distance_To_Hydrology
		data_train.rename({'Vertical_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		data_test.rename({'Vertical_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		#on supprime la colonne 
		data_train.drop(["Horizontal_Distance_To_Hydrology"],axis = 1, inplace = True)		 
		data_test.drop(["Horizontal_Distance_To_Hydrology"],axis = 1, inplace = True)
		

		#Etape 3 : Regroupement des variables Hillshade

		#fusion de Hillshade_9am et Hillshade_3pm : addition
		data_train['Hillshade_9am'] = data_train['Hillshade_9am'] + data_train['Hillshade_3pm']
		data_train.rename({'Hillshade_9am':'Hillshade_fusion'}, axis = 1, inplace = True)
		data_train.drop(["Hillshade_3pm"], axis = 1, inplace = True)
		#print(data["Hillshade_fusion"])

		data_test['Hillshade_9am'] = data_test['Hillshade_9am'] + data_test['Hillshade_3pm']
		data_test.rename({'Hillshade_9am':'Hillshade_fusion'}, axis = 1, inplace = True)
		data_test.drop(["Hillshade_3pm"], axis = 1, inplace = True)
		

		"""
		data_train[4] = np.sqrt(np.square(data_train[3] + np.square(data_train[4])))
		data_test[4] = np.sqrt(np.square(data_test[3] + np.square(data_test[4])))
		#on renomme la fusion des deux en VH_Distance_To_Hydrology
		data_train.rename({'Vertical_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		data_test.rename({'Vertical_Distance_To_Hydrology':'VH_Distance_To_Hydrology'}, axis = 1, inplace = True)
		#on supprime la colonne 
		data_train.drop([3],axis = 1, inplace = True)		 
		data_test.drop([3],axis = 1, inplace = True)
		"""
		return data_train, data_test, target_train, target_test #on revoit data au final 

############################ Etape 4 : Méthodes d'apprentissage ##################################

class Apprentissage: 
	def Naive_Bayes(data):
		print("Méthode de Gauss : Classification Naive Bayes, suppose que chaque classe est contruite à partir d'une distribution Gaussienne alignée. Avantage : très rapide.")
		#classifieur 
		classifier = GaussianNB()
		#apprentissage 
		classifier.fit(data_train, target_train)
		#Exécution de la prédiction sur les données d'apprentissage
		target_pred = classifier.predict(data_test) #résultats obtenus
		# qualité de la prédiction
		print("Qualité de la prédiction : ",accuracy_score(target_pred, target_test))
	
	def KNN(data,data_train, data_test, target_train, target_test):
		classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
		classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def perceptron_multi_couches(data,data_train, data_test, target_train, target_test): 
		classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001) 
		classifier.fit(data_train, target_train)
		target_pred = classifier.predict(data_test)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def arbre_de_decision(data,data_train, data_test, target_train, target_test): 
		#pour entrainer et prendre des decisions
		classifier = DecisionTreeClassifier()
		classifier.fit(data_train, target_train)
		#pour faire des prédictions 
		target_pred = classifier.predict(data_test)
		conf = confusion_matrix(target_test, target_pred)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

	def random_forest(data,data_train, data_test, target_train, target_test):
		classifier = RandomForestClassifier(n_estimators=30,criterion='entropy',max_features=None)
		classifier.fit(data_train,target_train)
		target_pred = classifier.predict(data_test)
		#Pretraitement.matrice_de_confusion(target_test,target_pred)
		print("Qualité de la prédiction : ", accuracy_score(target_test, target_pred))

################################################################## Appel de fonctions ##################################################################################

############################## Etape 1 : manipulation des données ##################################

fichier = "covtype.data"
fichier_modifie = "covtype_modifie.data"

Annexe.affichage("lecture_fichier_Numpy") 
data_np = Lecture.lecture_fichier_Numpy(fichier)
print("lecture finie")
print("\n")

Annexe.affichage("lecture_fichier_Pandas") 
data_pandas = Lecture.lecture_fichier_Pandas(fichier_modifie)
print("lecture finie")
print("\n")
"""
Annexe.affichage("moyenne")
Manipulation_donnees.moyenne(data_np)
print("\n")

Annexe.affichage("type_foret") 
Manipulation_donnees.type_foret(data_np)
print("\n")

############################## Etape 2 : analyse des données ##################################

Annexe.affichage("analyse_usuelle") 
element = 0 #0 = Elevation, 1 = Aspect, 2 = Slope ....
Analyse_de_donnees.analyse_usuelle(data_pandas,element)
print("\n")

Annexe.affichage("croisement_de_variables") 
Analyse_de_donnees.croisement_de_variables(data_pandas)
print("\n")

Annexe.affichage("histogramme") 
type_element = 'Horizontal_Distance_To_Fire_Points'
show = True
Analyse_de_donnees.histo(data_pandas,type_element,show)
print("\n")

Annexe.affichage("boxplot") 
type_element='Elevation'
show = True 
Analyse_de_donnees.boxplot(data_pandas,type_element,show)
print("\n")

Annexe.affichage("ACP")
Analyse_de_donnees.ACP(data_pandas)
print("\n")

Annexe.affichage("correlation_variables")
Analyse_de_donnees.correlation_variables(data_np)
print("\n")

"""

############################## Etape 3 : pré-traitements et construction des descripteurs ##################################

Annexe.affichage("separation_donnees")
Pretraitement.separation_donnees(data_np)
print("\n")

Annexe.affichage("epuration_donnees")
data_train,data_test,target_train,target_test = Pretraitement.epuration_donnees(data_pandas)
print("\n")


############################ Etape 4 : Méthodes d'apprentissage ##################################


print("Résultats avant épuration :")

data_train, data_test, target_train, target_test = Pretraitement.separation_donnees(data_np)

"""
Annexe.affichage("Naive_Bayes")
Apprentissage.Naive_Bayes(data_np,data_train, data_test, target_train, target_test)
print("\n")

Annexe.affichage("KNN")
Apprentissage.KNN(data_np,data_train, data_test, target_train, target_test)
print("\n")

Annexe.affichage("perceptron_multi_couches")
Apprentissage.perceptron_multi_couches(data_np,data_train, data_test, target_train, target_test)
print("\n")

Annexe.affichage("arbre_de_decision")
Apprentissage.arbre_de_decision(data_np,data_train, data_test, target_train, target_test)
print("\n")

Annexe.affichage("random_forest")
Apprentissage.random_forest(data_np,data_train, data_test, target_train, target_test)
print("\n")
"""


print("Résultats après épuration :")

Annexe.affichage("arbre_de_decision")
Apprentissage.arbre_de_decision(data_pandas,data_train,data_test,target_train,target_test)
print("\n")

"""
Annexe.affichage("random_forest")
Apprentissage.random_forest(data_epuree)
print("\n")
"""

#A faire :) =  
#1) fonction epuration des données = 
# - rendre la disparité entre les classes moins importantes (genre trop d'éléments dans la classe 1 par rapport aux autres classes etc)
# - fusionner vertical_distance_to_hydrology et horizontal_distance_to_hydrology, voir pour les autres variables  
#2) lancement de fonctions = 
#lancer les deux donctions qui suivent avec data_epuree, data retourné par la fonction epuration des données 