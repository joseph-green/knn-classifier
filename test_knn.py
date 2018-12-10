import csv
import knn.knn as knn
import sys

def getClass(position):
	#take the first character to avoid troubles with players with multiple positions (e.g. L/C)
	if position[0] == "D":
		return "Defense"
	elif position[0] == "G":
		return "Goalie"
	else:
		return "Forward"

# creates a new file with properly formatted data
def cleanData(filename):

	with open(filename[:-4]+"_formatted.csv",mode='w',newline='') as newcsvfile:
		csvWriter = csv.writer(newcsvfile)
		csvWriter.writerow(["gamesPlayed","goals","assists","points","penaltyMinutes","class"])

		with open(filename,mode='r') as csvfile:
			csvReader = csv.DictReader(csvfile)
			for row in csvReader:
				formattedData = [row["GP"],row["G"],row["A"],row["Pts"],row["PIM"],getClass(row["pos"])]
				csvWriter.writerow(formattedData)

cleanData(sys.argv[1])

classifier = knn.KNNClassifier(sys.argv[1][:-4]+"_formatted.csv",k=7)

cleanData(sys.argv[2])

with open(sys.argv[2][:-4]+"_formatted.csv") as testfile:
	
	testReader = csv.reader(testfile)
	
	successes = 0
	failures = 0
	unclassified = 0
	
	next(testReader) # ignore header
	
	for row in testReader:

		actualResult = row[-1]

		try:
		
			knnResult = classifier.classify(row[:-1])
		
		except ValueError:
		
			unclassified += 1
		
			continue

		if actualResult == knnResult:
		
			print("Player correctly classified as " + knnResult + "\n")
		
			successes += 1
		
		else:
		
			print(actualResult + " incorrectly identified as " + knnResult + "\n")
		
			failures += 1
	
	successRate = (successes/(successes+failures)) * 100
	print("Success Rate: " + str(successRate) + "% (" + str(successes+failures) + " out of " + str(successes+failures+unclassified) + " entries classified)\n" )




