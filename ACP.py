#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# classe pour standardisation (ACP)
from sklearn.preprocessing import StandardScaler
# classe pour l'ACP
from sklearn.decomposition import PCA
# Cercle de corrélation
from mpl_toolkits.mplot3d import Axes3D

def lecture_fichier_Pandas(fichier):
	data = pandas.read_table("covtype_modifie.data",sep = ',',header = 0)
	return data

def lecture_fichier_Numpy(fichier): 
	data = np.loadtxt("covtype.data",delimiter=",") #autre type d'utilisation des données 
	return data

def affichage(nom):
	print "################################ Fonction",nom," : #########################################"

def ACP(data):
	# Nous devons explicitement centrer et réduire les données pour réaliser une ACP normée avec PCA
	# Instanciation
	sc = StandardScaler()
	Z = sc.fit_transform(data)
	# On ne garde que les variables quantitatives
	Z = Z[:,0:9]
	# On peut vérifier les propriétés de ces nouvelles données
	# Moyenne
	print "Le tableau des moyennes est : ", np.mean(Z,axis=0)
	print "Aux erreurs de troncature près, les moyennes valent 0"
	# Ecart-type
	print "Le tableau des ecart-type est : ", np.std(Z,axis=0,ddof=0)

	# Nous pouvons maintenant faire l'ACP
	# Instanciation
	acp = PCA(svd_solver='full')
	# Affichage des paramètres
	print(acp)

	#calculs
	coord = acp.fit_transform(Z)


	# On peut utiliser la "cassure" pour déterminer le nombre de facteurs à retenir
	p = acp.n_components_
	print "Avec les 3 premières variables, on a 70 pourcents de l'information"
	print "Avec les 4 premières variables, on a 80 pourcents de l'information"
	print "Avec les 6 premières variables, on a 90 pourcents de l'information"

	plt.plot(np.arange(1,p+1),np.cumsum(acp.explained_variance_ratio_))
	plt.title("Explained variance vs. # of factors")
	plt.ylabel("Cumsum explained variance ratio")
	plt.xlabel("Factor number")
	plt.show()
	n = Z.shape[0]
	eigval = acp.singular_values_**2/n

	# 2e méthode : Test des bâtons brisés
	#seuils pour test des bâtons brisés
	n = Z.shape[0]
	bs = np.zeros(p)
	compteur = 0
	for i in range(p):
		bs[i] = 1.0/float(p-i)
	bs = np.cumsum(bs)
	bs = bs[::-1]
	print(pandas.DataFrame({'Val.Propre':eigval,'Seuils':bs}))

	# Cercle de corrélation - Incompréhensible
	'''
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],cmap=plt.cm.Paired)
	ax.set_title("ACP: trois premieres composantes")
	ax.set_xlabel("Comp1")
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("Comp2")
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("Comp3")
	ax.w_zaxis.set_ticklabels([])
	plt.show()
	#positionnement des individus dans le premier plan
	fig, axes = plt.subplots(figsize=(12,12))
	axes.set_xlim(-6,6) 
	#même limites en abscisse
	axes.set_ylim(-6,6) 
	#et en ordonnée
	#placement des étiquettes des observations
	for i in range(n):
		plt.annotate(i,(coord[i,0],coord[i,1]))
	#ajouter les axes
	plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
	plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
	#affichage
	plt.show()'''

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
	print "On a bien la somme des cos égale à 1"

	#contributions aux axes
	ctr = coord**2
	for j in range(p):
		ctr[:,j] = ctr[:,j]/(n*eigval[j])
	print(pandas.DataFrame({'id':range(n),'CTR_1':ctr[:,0],'CTR_2':ctr[:,1],'CTR_2':ctr[:,2]}))

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

###################### Appel de fonctions #########################

fichier = "covtype.data"
fichier_modifie = "covtype_modifie.data"

affichage("lecture_fichier_Numpy") 
data_np = lecture_fichier_Numpy(fichier)
print("\n")

'''
affichage("lecture_fichier_Pandas") 
data_pandas = lecture_fichier_Pandas(fichier_modifie)
print("\n")
'''

affichage("ACP")
ACP(data_np)
print("\n")