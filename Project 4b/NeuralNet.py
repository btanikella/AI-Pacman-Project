import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

#Author: Georgia Tech
#Edited: Bharadwaj Tanikella

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        denominator = 1+exp(-1*value)
        return 1/denominator
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as rounded g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The rounded value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        inActs.insert(0,1)
        return round(self.sigmoid(self.getWeightedSum(inActs)))
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        numerator = exp(value)
        denominator = ((1+exp(value))*(1+exp(value)))
        return numerator/denominator
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        temp=inActs[:]
        temp.insert(0,1)
        weightedSum = self.getWeightedSum(temp)
        return self.sigmoidDeriv(weightedSum)
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        """YOUR CODE"""
        returnList = []
        tempList = inActs[:]
        tempList.insert(0,1) #add the bias for inActs 

        for i in range(len(tempList)):
            weight = self.weights[i]
            tempWeight = self.weights[i]+(alpha*tempList[i]*delta)
            self.weights[i]=tempWeight
            modification = tempWeight-weight
            returnList.append(abs(modification))
        return sum(returnList)
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        """YOUR CODE"""
        tempList = inActs[:]
        returnList = []
        returnList.append(tempList) #list of lists
        count = 0
        count2=0

        for i in range(self.numHiddenLayers):#Iterate through the Hidden Layers.
            curr= []
            count+=1
            for l in self.hiddenLayers[i]:
                curr.append(l.sigmoidActivation(tempList[:]))
                count2+=1
            returnList.append(curr)
            tempList= curr[:]

        finalList=[]

        for j in self.outputLayer:
            finalList.append(j.sigmoidActivation(tempList[:]))
        returnList.append(finalList)

        return returnList
    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        
        Args: 
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons, 
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """
        #keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0
        
        
        #Dictionaries to keep track of the error List and the Weights of the Layers. They are mainly present to keep track of values outside the for loop.
        tempWeights = []
        errList=[]
        tempValue = 0


        for example in examples:#for each example
            deltas = []#keep track of deltas to use in weight change
            
            """YOUR CODE"""
            """Get output of all layers"""
            allLayerOutput = self.feedForward(example[0])

            """
            Calculate output errors for each output perceptron and keep track 
            of error sum. Add error delta values to list.
            """
            outPutError= 0
            temp=[]

            count=0
            lengthLayer= len(allLayerOutput)- 1

            for i in allLayerOutput[lengthLayer]:
                error = example[1][count]-i
                error = (error*error)/2
                tempVal= self.outputLayer[count].sigmoidActivationDeriv(allLayerOutput[lengthLayer-1]) * (example[1][count]-i)
                temp.append(tempVal)
                outPutError += error
                count += 1
            deltas.append(temp)
        
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            Be careful to account for bias inputs! 
            """
            tempLe= len(self.hiddenLayers)-1

            for i in range(tempLe,-1,-1): #reverse Iteration
                tempList=[]
                countA = 0
                for l in self.hiddenLayers[i]:
                    someVal = l.sigmoidActivationDeriv(allLayerOutput[i])
                    weights=[]
                    if(i==self.numHiddenLayers-1):
                        for n in self.outputLayer:
                            weights.append(n.weights[countA+1])
                    else:
                        for n in self.hiddenLayers[i+1]:
                            weights.append(n.weights[countA+1])
                    tempList.append(sum([a*b for a,b in zip(weights,deltas[0])]) * someVal)
                    countA+=1
                deltas.insert(0,tempList[:])

            """
            Having aggregated all deltas, update the weights of the 
            hidden and output layers accordingly.
            """      
        #end for each example

            finalDeltas = [val for sub in deltas for val in sub] #double Iteration. 

            counter = 0

            allLayerCounter = 0
            
            #iteration through the hiddenLayers
            for l in self.hiddenLayers:

                #Iteration through the layer to update the values.
                for i in l:

                    tempWeights.append((i.updateWeights(allLayerOutput[allLayerCounter],alpha,finalDeltas[counter])))

                    counter+=1

                    tempValue+=(len(allLayerOutput[allLayerCounter])+1)

                allLayerCounter+=1
          
            #iteration through the outputLayer
            for i in self.outputLayer:
                tempWeights.append((i.updateWeights(allLayerOutput[allLayerCounter],alpha,finalDeltas[counter])))
                counter+=1
                tempValue+=(len(allLayerOutput[len(allLayerOutput)-2])+1)   

            errList.append(outPutError)
 
        averageWeightChange = sum(tempWeights)/tempValue
        averageError = sum(errList)/(len(examples)*len(self.outputLayer))


        """Calculate final output"""
        return averageError, averageWeightChange
    
def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration=0
    trainError=0
    weightMod=0

    #First Iteration. 
    iteration = 1
    weightMod = nnet.backPropLearning(examplesTrain, alpha)[1]
    
    while (weightMod>weightChangeThreshold and iteration<maxItr):
        tempWeight = nnet.backPropLearning(examplesTrain, alpha)
        trainError = tempWeight[0] 
        weightMod = tempWeight[1]
        iteration+=1
    
    """
    Iterate for as long as it takes to reach weight modification threshold
    """
        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',
        
          
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
    """ 
    
    testError = 0
    testGood = 0     
    
    testAccuracy=0#num correct/num total
    
    for inp,out in examplesTest:
        tempFF= nnet.feedForward(inp)
        current=[]
        values=tempFF[len(tempFF)-1]

        #Change the list to float
        answer = [float(i) for i in out]

        counter=0
        for i in range(len(answer)):
            if(answer[i]==values[i]):
                counter+=1

        if(counter==len(answer)):
            testGood +=1
        else:
            testError+=1

    testAccuracy = testGood/((testGood+testError)*1.0) 
    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    
    """return something"""
    return nnet,testAccuracy

