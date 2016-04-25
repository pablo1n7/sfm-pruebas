import cv2
import numpy as np
class Feature(object):

	puntos_uno = None
	puntos_dos = None
	match_puntos_uno = None
	match_puntos_dos = None
	vista_uno = None
	vista_dos = None
	descriptor_uno = None
	descriptor_dos = None


	"""docstring for Feature"""
	def __init__(self, vista1,vista2):
		
		self.vista_uno = vista1
		self.vista_dos = vista2		
		surf = cv2.SURF()
		self.puntos_uno, self.descriptor_uno = surf.detectAndCompute(self.vista_uno.img,None)
		self.puntos_dos, self.descriptor_dos = surf.detectAndCompute(self.vista_dos.img,None)
		
		matcher = cv2.BFMatcher(cv2.NORM_L1, True)
		matches = matcher.match(self.descriptor_uno, self.descriptor_dos)

		# generate lists of point correspondences
		self.match_puntos_uno = np.zeros((len(matches), 2), dtype=np.float32)
		self.match_puntos_dos = np.zeros_like(self.match_puntos_uno)

		for i in range(len(matches)):
		    self.match_puntos_uno[i] = self.puntos_uno[matches[i].queryIdx].pt
		    self.match_puntos_dos[i] = self.puntos_dos[matches[i].trainIdx].pt