from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

# Question 6 Data
listMAS = []
for j in range(0,45,5): #for car data iterations
    penDataList = []
    carDataList = []
    for i in range(5):
        b = testCarData([j])
        carDataList.append(b[1])
    aver2 = average(carDataList)
    max2 = max(carDataList)
    stddev2 = stDeviation(carDataList)
    v = (aver2,max2,stddev2)
    print v
    listMAS.append((aver2,max2,stddev2))
    listMAS.append((0,0,0))
  
list2 = []   
for j in range(0,45,5): #for pen data iterations
    penDataList = []
    carDataList = []
    for i in range(5):
        a = testPenData([j])
        penDataList.append(a[1])
    aver1 = average(penDataList)
    max1 = max(penDataList)
    stddev1 = stDeviation(penDataList)
    u = (aver1,max1,stddev1)
    print u
    list2.append((aver1,max1,stddev1))
print list2