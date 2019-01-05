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

############################## Etape 1 : manipulation des données ##################################

Annexe.affichage("lecture_fichier_Pandas") 
data_pandas = Lecture.lecture_fichier_Pandas_modifie()
print("lecture finie")
print("\n")

Annexe.affichage("moyenne")
Manipulation_donnees.moyenne()
print("\n")

"""
Annexe.affichage("type_foret") 
Manipulation_donnees.type_foret(data_np)
print("\n")
"""

"""
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

############################## Etape 3 : pré-traitements et construction des descripteurs ##################################

Annexe.affichage("separation_donnees")
data_train, data_test, target_train, target_test = Pretraitement.separation_donnees(data_pandas)
print("\n")

############################ Etape 4 : Méthodes d'apprentissage ##################################

############################ Résultat avant épuration ############################

print("Résultats avant épuration :")
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
Apprentissage.arbre_de_decision(data_train, data_test, target_train, target_test)
print("\n")

Annexe.affichage("random_forest")
Apprentissage.random_forest(data_train, data_test, target_train, target_test)
print("\n")

############################ Résultat après épuration ############################

print("Résultats après épuration :")
Annexe.affichage("epuration_donnees")
data_train,data_test,target_train,target_test = Pretraitement.epuration_donnees(data_pandas)
print("data_train",data_train)
print("\n")

Annexe.affichage("arbre_de_decision")
Apprentissage.arbre_de_decision(data_train,data_test,target_train,target_test)
print("\n")

Annexe.affichage("random_forest")
Apprentissage.random_forest(data_train,data_test,target_train,target_test)
print("\n")
"""

#A faire :) =  
#1) fonction epuration des données = 
# - rendre la disparité entre les classes moins importantes (genre trop d'éléments dans la classe 1 par rapport aux autres classes etc)
# - fusionner vertical_distance_to_hydrology et horizontal_distance_to_hydrology, voir pour les autres variables  
