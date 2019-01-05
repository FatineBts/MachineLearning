#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np

############################## Etape 1 : manipulation des données ##################################

class Lecture: 
	#lecture des fichiers
	def lecture_fichier_Numpy(fichier): 
		data = np.loadtxt(fichier,delimiter=",") #autre type d'utilisation des données 
		return data

	def lecture_fichier_Pandas(fichier):
		data = pandas.read_csv(fichier,delimiter=",",header = 0)
		return data