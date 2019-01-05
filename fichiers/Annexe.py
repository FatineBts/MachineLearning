#!/usr/bin/env python
# -*- coding: utf-8 -*-

############################## Etape 1 : manipulation des données ##################################

class Annexe: 
	def nombre_de_lignes(data): 
		nombre_de_lignes = 0
		nombre_lignes,colonne = data.shape
		return nombre_lignes

	def affichage(nom):
		print("################################ Fonction",nom," : #########################################")
