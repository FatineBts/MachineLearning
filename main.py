#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

from fichiers.Manipulation_donnees import *
from fichiers.Annexe import *
from fichiers.Lecture import *
from fichiers.Analyse_de_donnees import *
from fichiers.Pretraitement import *
from fichiers.Apprentissage import *

################################################################## Appel de fonctions ##################################################################################

print("Tapez 1 pour lancer la partie Manipulation_donnees\nTapez 2 pour lancer la partie Analyse_de_donnees\nTapez 3 pour le pré-traitement + méthodes d'apprentissage\nTapez 4 pour tout lancer : ")

r = input()

############################## Etape 1 : manipulation des données ##################################

Annexe.affichage("lecture_fichier_Pandas") 
data_pandas = Lecture.lecture_fichier_Pandas()
print("lecture finie")
print("\n")

if(r=='1' or r=='4'):
	Annexe.affichage("moyenne")
	Manipulation_donnees.moyenne(data_pandas)
	print("\n")

	Annexe.affichage("type_foret") 
	Manipulation_donnees.type_foret(data_pandas)
	print("\n")

############################## Etape 2 : analyse des données ##################################

if(r=='2' or r=='4'):
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
	Analyse_de_donnees.correlation_variables(data_pandas)
	print("\n")

############################## Etape 3 : pré-traitements et construction des descripteurs ##################################

if(r=='3' or r=='4'):

	Annexe.affichage("separation_donnees")
	data_train, data_test, target_train, target_test = Pretraitement.separation_donnees(data_pandas)
	print("\n")

	Annexe.affichage("epuration_donnees")
	data_train,data_test,target_train,target_test = Pretraitement.epuration_donnees(data_pandas)
	print("\n")

############################ Etape 4 : Méthodes d'apprentissage ##################################

	Annexe.affichage("Naive_Bayes")
	Apprentissage.Naive_Bayes(data_train, data_test, target_train, target_test)
	print("\n")

	Annexe.affichage("KNN")
	Apprentissage.KNN(data_train, data_test, target_train, target_test)
	print("\n")

	Annexe.affichage("perceptron_multi_couches")
	Apprentissage.perceptron_multi_couches(data_train, data_test, target_train, target_test)
	print("\n")

	Annexe.affichage("arbre_de_decision")
	Apprentissage.arbre_de_decision(data_train,data_test,target_train,target_test)
	print("\n")

	Annexe.affichage("random_forest")
	Apprentissage.random_forest(data_train,data_test,target_train,target_test)
	print("\n")
