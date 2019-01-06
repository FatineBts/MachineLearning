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
	def lecture_fichier_Pandas():
		data = pandas.read_csv("covtype.data",delimiter=",",header = 0)
		return data