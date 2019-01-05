#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fichiers.Annexe as Annexe 
import numpy as np
import matplotlib.pyplot as plt

############################## Etape 1 : manipulation des données ##################################

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

