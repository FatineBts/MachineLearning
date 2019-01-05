#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
#Polytech Sorbonne - année 2018/2019
#Réalisé par : Fatine Bentires Alj et Alexia Zounias-Sirabella
#Cours d'apprentissage statistique de Patrick Gallinari
##################################################################

############################## Etape 1 : manipulation des données ##################################

class Annexe: 
	def nombre_de_lignes(data): 
		nombre_de_lignes = 0
		nombre_lignes,colonne = data.shape
		return nombre_lignes

	def affichage(nom):
		print("################################ Fonction",nom," : #########################################")
