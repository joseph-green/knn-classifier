import csv
import math
import statistics
import itertools

class KNNClassifier():
	
	def __init__(self,filename,k=1):
		
		self.datafile = filename
		
		self.k = k


	# classifies toClassify as one of the classes indicated in the datafile, or raises a ValueError if no decision can be made 
	def classify(self,toClassify):
		
		data = csv.reader(open(self.datafile,newline=''))
		
		# values in nearestNeighbours are stored as [classification,distance], and entries are sorted by distance
		# we start by taking the first k elements in the training data
		nearestNeighbours = [[row[-1],self.getDistance(toClassify,row[:-1])] for row in itertools.islice(data,1,self.k+1)]
		nearestNeighbours.sort(key=lambda x : x[1])


		for entry in itertools.islice(data,self.k,0): # since first k elements are in nearestNeighbours, start at (k+1)th index

			totalDistance = self.getDistance(toClassify,entry[:-1]) # omit the last data point (the class)

			if (len(nearestNeighbours) < self.k or totalDistance < nearestNeighbours[-1][1]):

				insertPosition = len(nearestNeighbours) - 1

				while (totalDistance < nearestNeighbours[insertPosition-1][1] and insertPosition > 0):

					insertPosition -= 1

				nearestNeighbours[insertPosition] = [entry[-1],totalDistance]

		results = [neighbour[0] for neighbour in nearestNeighbours] # generate a list with just the classes

		try:

			return statistics.mode(results)

		except statistics.StatisticsError: # raises this error when no mode can be found

			raise ValueError("Could not classify: multiple classes contain the same number of neighbours")

	
	# returns the Euclidean distance of two data entries
	def getDistance(self,entry1,entry2):

		distance = 0

		for val1,val2 in zip(entry1,entry2):
		
			distance += (int(val1) - int(val2)) ** 2
		
		return math.sqrt(distance)