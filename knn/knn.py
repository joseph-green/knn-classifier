import csv
import math
import statistics
import itertools
import numpy as np

class KNNClassifier():
	
	def __init__(self,datafile,k=1,weights=None):
		
		self.__datafile = datafile

		self.__k = k

		reader = csv.reader(open(datafile,newline=''))
		header = next(reader)

		#if weighting is specified, set it to 
		if (weights):
			self.weights = weights
		else:
			self.weights = [(1/len(header)) for n in header]


	@property
	def k(self):
		return self.__k

	@k.setter
	def setK(self, newK):

		if not newK.is_integer():
			raise ValueError("Invalid k-value: must be an is_integer")
		elif newK <= 0:
			raise ValueError("Invalid k-value: must be greater than 0")
		else:
			self.__k = newK


	def __normalize(self,data):
		for i in range(0,data.shape[1]-1): # iterate through indexes 0 .. k-2 to exclude class
			data[:,i] -= data[:,i].min()
			data[:,i] /= data[:,i].max()

	# classifies toClassify as one of the classes indicated in the datafile, or raises a ValueError if no decision can be made 
	def classify(self,toClassify):
		
		data = np.genfromtxt(self.__datafile,names=True,delimiter=",",dtype=None,encoding=None)
		

		#store the class and variables in seperate lists since classes contains a string and data contains numeric data
		classes = np.asarray([t[-1] for t in data])
		data = np.asarray([list(t)[:-1] for t in data],dtype=float)
		
		#data = data.reshape(data.shape[0],1)
		#data = np.asarray(data)
		self.__normalize(data)
		#data = csv.reader(open(self.datafile,newline=''))
		
		# values in nearestNeighbours are stored as [classification,distance], and entries are sorted by distance
		# we start by taking the first k elements in the training data
		#nearestNeighbours = [[row[-1],self.getDistance(toClassify,row[:-1])] for row in itertools.islice(data,1,self.k+1)]
		nearestNeighbours = [[classes[i],self.__getDistance(toClassify,data[i])] for i in range(0,self.k)]
		nearestNeighbours.sort(key=lambda x : x[1])


		for clss, datapoint in itertools.islice(zip(classes,data),self.k,0): # since first k elements are in nearestNeighbours, start at (k+1)th index

			totalDistance = self.getDistance(toClassify,datapoint) # omit the last data point (the class)

			if (len(nearestNeighbours) < self.k or totalDistance < nearestNeighbours[-1][1]):

				insertPosition = len(nearestNeighbours) - 1

				while (totalDistance < nearestNeighbours[insertPosition-1][1] and insertPosition > 0):

					insertPosition -= 1

				nearestNeighbours[insertPosition] = [clss,totalDistance]

		results = [neighbour[0] for neighbour in nearestNeighbours] # generate a list with just the classes

		try:

			return statistics.mode(results)

		except statistics.StatisticsError: # raises this error when no mode can be found

			raise ValueError("Could not classify: multiple classes contain the same number of neighbours")


	# classify multiple entries in a .csv file
	def classifyFile(filename):

		data = csv.reader(open(filename,newline=''))

		for entry in itertools.islice(data,1):

			self.classify(entry)


	# returns the Euclidean distance of two data entries
	def __getDistance(self,entry1,entry2):

		distance = 0

		for val1, val2, weight in zip(entry1,entry2,self.weights):
		
			distance += ((int(val1) - int(val2)) ** 2) * weight
		
		return math.sqrt(distance)
