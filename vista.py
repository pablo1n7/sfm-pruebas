from feature import *
import cv2

class Vista(object):
	
	"""Representa una imagen,con su correspondientes caracteristicas aplicadas a otra vista"""
	ruta = ""
	features = None
	img = []

	def __init__(self, ruta, size=600):
		self.ruta = ruta
		self.img = cv2.imread(ruta,cv2.CV_8UC3)#if len(self.img.shape) == 2:
		self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
		if self.img.shape[1] > size:
			while self.img.shape[1] > 2*size:
				self.img = cv2.pyrDown(self.img)

		self.features = dict()

	def buscar_feature(self,vista2,matcher):
		feacture = Feature(self,vista2,matcher)
		self.features[vista2] = feacture
		vista2.features[self] = feacture
	