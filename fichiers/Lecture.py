#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

import pandas
import numpy as np

############################## Etape 1 : manipulation des données ##################################

class Lecture: 
	#lecture des fichiers
	def lecture_fichier_Numpy(): 
		data = np.loadtxt("covtype.data",delimiter=",") #autre type d'utilisation des données 
		return data

	def lecture_fichier_Pandas_base(): 
		data = pandas.read_csv("covtype.data",delimiter=",",header = 0)
		return data
	
	def lecture_fichier_Pandas_modifie():
		data = pandas.read_csv("covtype_modifie.data",delimiter=",",header = 0)
		return data