from collections import deque

"""
	Base class for unary constraints
	Implement isSatisfied in subclass to use
"""
class UnaryConstraint:
	def __init__(self, var):
		self.var = var

	def isSatisfied(self, value):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var


"""	
	Implementation of UnaryConstraint
	Satisfied if value does not match passed in paramater
"""
class BadValueConstraint(UnaryConstraint):
	def __init__(self, var, badValue):
		self.var = var
		self.badValue = badValue

	def isSatisfied(self, value):
		return not value == self.badValue

	def __repr__(self):
		return 'BadValueConstraint (%s) {badValue: %s}' % (str(self.var), str(self.badValue))


"""	
	Implementation of UnaryConstraint
	Satisfied if value matches passed in paramater
"""
class GoodValueConstraint(UnaryConstraint):
	def __init__(self, var, goodValue):
		self.var = var
		self.goodValue = goodValue

	def isSatisfied(self, value):
		return value == self.goodValue

	def __repr__(self):
		return 'GoodValueConstraint (%s) {goodValue: %s}' % (str(self.var), str(self.goodValue))


"""
	Base class for binary constraints
	Implement isSatisfied in subclass to use
"""
class BinaryConstraint:
	def __init__(self, var1, var2):
		self.var1 = var1
		self.var2 = var2

	def isSatisfied(self, value1, value2):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var1 or var == self.var2

	def otherVariable(self, var):
		if var == self.var1:
			return self.var2
		return self.var1


"""
	Implementation of BinaryConstraint
	Satisfied if both values assigned are different
"""
class NotEqualConstraint(BinaryConstraint):
	def isSatisfied(self, value1, value2):
		if value1 == value2:
			return False
		return True

	def __repr__(self):
	    return 'NotEqualConstraint (%s, %s)' % (str(self.var1), str(self.var2))


class ConstraintSatisfactionProblem:
	"""
	Structure of a constraint satisfaction problem.
	Variables and domains should be lists of equal length that have the same order.
	varDomains is a dictionary mapping variables to possible domains.

	Args:
		variables (list<string>): a list of variable names
		domains (list<set<value>>): a list of sets of domains for each variable
		binaryConstraints (list<BinaryConstraint>): a list of binary constraints to satisfy
		unaryConstraints (list<UnaryConstraint>): a list of unary constraints to satisfy
	"""
	def __init__(self, variables, domains, binaryConstraints = [], unaryConstraints = []):
		self.varDomains = {}
		for i in xrange(len(variables)):
			self.varDomains[variables[i]] = domains[i]
		self.binaryConstraints = binaryConstraints
		self.unaryConstraints = unaryConstraints

	def __repr__(self):
	    return '---Variable Domains\n%s---Binary Constraints\n%s---Unary Constraints\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + '\n' for e in self.binaryConstraints]), \
	        ''.join([str(e) + '\n' for e in self.unaryConstraints]))


class Assignment:
	"""
	Representation of a partial assignment.
	Has the same varDomains dictionary stucture as ConstraintSatisfactionProblem.
	Keeps a second dictionary from variables to assigned values, with None being no assignment.

	Args:
		csp (ConstraintSatisfactionProblem): the problem definition for this assignment
	"""
	def __init__(self, csp):
		self.varDomains = {}
		for var in csp.varDomains:
			self.varDomains[var] = set(csp.varDomains[var])
		self.assignedValues = { var: None for var in self.varDomains }

	"""
	Determines whether this variable has been assigned.

	Args:
		var (string): the variable to be checked if assigned
	Returns:
		boolean
		True if var is assigned, False otherwise
	"""
	def isAssigned(self, var):
		return self.assignedValues[var] != None

	"""
	Determines whether this problem has all variables assigned.

	Returns:
		boolean
		True if assignment is complete, False otherwise
	"""
	def isComplete(self):
		for var in self.assignedValues:
			if not self.isAssigned(var):
				return False
		return True

	"""
	Gets the solution in the form of a dictionary.

	Returns:
		dictionary<string, value>
		A map from variables to their assigned values. None if not complete.
	"""
	def extractSolution(self):
		if not self.isComplete():
			return None
		return self.assignedValues

	def __repr__(self):
	    return '---Variable Domains\n%s---Assigned Values\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + ':' + str(self.assignedValues[e]) + '\n' for e in self.assignedValues]))



####################################################################################################


"""
	Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
	Do not assign value to var. Only check if this value would be consistent or not.
	If the other variable for a constraint is not assigned, then the new value is consistent with the constraint.
	You do not have to consider unary constraints, as those have already been taken care of.

	Args:
		assignment (Assignment): the partial assignment
		csp (ConstraintSatisfactionProblem): the problem definition
		var (string): the variable that would be assigned
		value (value): the value that would be assigned to the variable
	Returns:
		boolean
		True if the value would be consistent with all currently assigned values, False otherwise
"""
def consistent(assignment, csp, var, value):
	"""Question 1"""
	"""YOUR CODE HERE"""
	#True if the value would be consistent with all currently assigned Values
	#print csp.binaryConstraints
	#print const.affects(var), Boolean if it is affecting or not. 
	for const in csp.binaryConstraints:
		if(const.affects(var) and value== assignment.assignedValues[const.otherVariable(var)]):
			return False
	return True


"""
	Recursive backtracking algorithm.
	A new assignment should not be created. The assignment passed in should have its domains updated with inferences.
	In the case that a recursive call returns failure or a variable assignment is incorrect, the inferences made along
	the way should be reversed. See maintainArcConsistency and forwardChecking for the format of inferences.

	Examples of the functions to be passed in:
	orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
	selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
	inferenceMethod: noInferences, maintainArcConsistency, forwardChecking

	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
		orderValuesMethod (function<assignment, csp, variable> returns list<value>): a function to decide the next value to try
		selectVariableMethod (function<assignment, csp> returns variable): a function to decide which variable to assign next
		inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>): a function to specify what type of inferences to use
			InferenceMethod will return None if the assignment has no solution. Otherwise it will return a set of inferences made. (The set can be empty.)
	Returns:
		Assignment
		A completed and consistent assignment. None if no solution exists.

	Author: Bharadwaj Tanikella. 
"""
def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
	"""Question 1"""
	"""YOUR CODE HERE"""
	# procedure bt(c)
	#   if reject(P,c) then return
	#   if accept(P,c) then output(P,c)
	#   s goes tofirst(P,c)
	#   while s is not equal to do
 #    bt(s)
 #    s goes too next(P,s)

 	#Returns assignment if it is complete and consistent.
 	#var,value=inferenceMethod
 	#if(assignment.isComplete() and consistent(assignment,csp,var, value)):
 	#return assignment

 	if(assignment.isComplete()):
 		return assignment

 	#Recursive Function to call for each selectVariableMethod to assign next. 
 	variable = selectVariableMethod(assignment,csp) # variable returns
 	if(variable==None):
 		return None
 	else:
 		nextValues= orderValuesMethod(assignment,csp,variable)
 		for values in nextValues:
 			if(not consistent(assignment,csp,variable,values)): #checks if it is consistent for every values produced by the orderValuesMethod
 				continue
 			else:
 				inferenceValues= inferenceMethod(assignment,csp,variable,values)
 				if inferenceValues is None:
 					continue
 				else:
	 				assignment.assignedValues[variable]=values
	 				returningVariable = recursiveBacktracking(assignment,csp,orderValuesMethod,selectVariableMethod,inferenceMethod)
	 				if returningVariable==None:
	 					for var,val in inferenceValues:
	 						assignment.varDomains[var].add(val)	
	 				else: #terminal phase for a recursive function.
	 					return returningVariable
	 				assignment.assignedValues[variable]=None



	return None


"""
	Uses unary constraints to eleminate values from an assignment.

	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
	Returns:
		Assignment
		An assignment with domains restricted by unary constraints. None if no solution exists.
"""
def eliminateUnaryConstraints(assignment, csp):
	domains = assignment.varDomains
	for var in domains:
		for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
			for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
				domains[var].remove(value)
				if len(domains[var]) == 0:
					# Failure due to invalid assignment
					return None
	return assignment


"""
	Trivial method for choosing the next variable to assign.
	Uses no heuristics.
"""
def chooseFirstVariable(assignment, csp):
	for var in csp.varDomains:
		if not assignment.isAssigned(var):
			return var


"""
	Selects the next variable to try to give a value to in an assignment.
	Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		the next variable to assign

	#Author: Bharadwaj Tanikella
"""
def minimumRemainingValuesHeuristic(assignment, csp):
	nextVar = None
	domains = assignment.varDomains
	"""Question 2"""
	"""YOUR CODE HERE"""
	#assignment is the Input which contains the assigned values which can help determine the heuristic. 
	#csp= Problem
	#nextVar = Dictionary to keep track of the degree heuristic winning variables
	#domains= Can be used to get the variable Domains[Range]
	varDict= domains.keys()
	length=len(varDict)
	#minimumDomainVariables=defaultdict()
	minimumDomainVariables =[]

	#Update the dictionary of minimumDomainVariables with recurrence to a domainValue.
	#Most Constrained Heuristic or Fail First Heuristic 
	#Degree needs to be assigned, Highest degree of variables available
	#Choose Highest Degree first, Prune the least constraining value.
	tempValue= len(domains)+1 #value to keep track of the domain space.
	for i in range(length):
		varLength= len(domains[varDict[i]])
		variable= varDict[i]
		if((assignment.assignedValues[variable]==None) and (tempValue > varLength)):
			minimumDomainVariables=[] #delete the array to start new. 
			tempValue=varLength
			minimumDomainVariables.append(variable)
		elif(assignment.assignedValues[variable]==None and (tempValue == varLength)):
			minimumDomainVariables.append(variable)

	#Dictionary minimumDomainVariables is filled.
	dictLength= len(minimumDomainVariables) #Length of the variables.

	#minimumDomainVariables first element is returned when dictLength is only one.
	if(dictLength==1):
		nextVar= minimumDomainVariables[0]

	#if dictLength is greater than 1 further degree is matched. 
	if(dictLength>1):
		variableDegree=0
		for variables in minimumDomainVariables:
			tempCount=0
			if(len(csp.binaryConstraints)==0):
				nextVar=minimumDomainVariables[0]
			else:
				for const in csp.binaryConstraints:
					if (const.affects(variables)):
						tempCount += 1
				if variableDegree < tempCount:
					variableDegree= tempCount
					nextVar= variables
	return nextVar


"""
	Trivial method for ordering values to assign.
	Uses no heuristics.
"""
def orderValues(assignment, csp, var):
	return list(assignment.varDomains[var])


"""
	Creates an ordered list of the remaining values left for a given variable.
	Values should be attempted in the order returned.
	The least constraining value should be at the front of the list.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable to be assigned the values
	Returns:
		list<values>
		a list of the possible values ordered by the least constraining value heuristic
"""
def leastConstrainingValuesHeuristic(assignment, csp, var):
	values = list(assignment.varDomains[var])
	"""Hint: Creating a helper function to count the number of constrained values might be useful"""
	"""Question 3"""
	"""YOUR CODE HERE"""
# 	Given a variable, choose the least constraining value:
# the one that rules out the fewest values in the remaining variables
# Idea is to explore paths, those with most options, most likely to lead to a solution.
	#print values
	answer = helperfunction(assignment,csp,var,values)
	values=[]
	for val in answer:
		values.append(val[0])

	return values

#Takes in the values of an assignment csp var values and returns a list[array] with a tupule of variable and number count of constrained values
def helperfunction(assignment,csp,var,values):
	constrainedValue=[]
	#Update the countValues to the variables
	for val in values:
		constraintValue=0
		for const in csp.binaryConstraints:
			if(const.affects(var)):
				variables= const.otherVariable(var)
				domain=list(assignment.varDomains[variables])
				if val in domain:
					constraintValue+=1
		constrainedValue.append((val,constraintValue))
	#return constrainedValue Fails test_cases/q3/leastconstrainingvalue2lcv.test Sorting is needed
	return sorted(constrainedValue, key=lambda tup: tup[1]) 

"""
	Trivial method for making no inferences.
"""
def noInferences(assignment, csp, var, value):
	return set([])


"""
	Implements the forward checking algorithm.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the fuction.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
	#Author Bharadwaj Tanikella
"""
def forwardChecking(assignment, csp, var, value):
	inferences = set([])
	domains = assignment.varDomains
	"""Question 4"""
	"""YOUR CODE HERE"""
	# Assume all constraints are Binary on variables X, and Y. 
	# constraint.get_X gives the X variable 
	# constraint.get_Y gives the Y variable 
	# X.domain gives the current domain of variable X. 
	# forward_checking( state ) 
	# 	run basic constraint checking fail if basic check fails
	# 	Let X be the current variable being assigned,
	# 	Let x be the current value being assigned to X.
	# 	check_and_reduce( state, X=x ) 
	visited = []
	constrainedVar=[]
	tempList=[]
	listLength=0
	val=value

	#Constraints are checked for the value and updated with inferences with the variable and value. 
	for const in csp.binaryConstraints:
		if(const.affects(var)):
			if(assignment.assignedValues[const.otherVariable(var)]==None):
				constrainedVar.append(const.otherVariable(var))
				variable = const.otherVariable(var)
				tempList = list(domains[variable])
				listLength= len(domains[variable])
				if val in tempList:
					if (listLength != 1):
						domains[variable].remove(val)
						inferences.add((variable,val))
						visited.append(variable)	
					else: #if the number of domain variables are 1
						for visit in visited:
							assignment.varDomains[visit].add(val)
						return None					
			
	return inferences


"""
	Helper funciton to maintainArcConsistency and AC3.
	Remove values from var2 domain if constraint cannot be satisfied.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the fuction.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var1 (string): the variable with consistent values
		var2 (string): the variable that should have inconsistent values removed
		constraint (BinaryConstraint): the constraint connecting var1 and var2
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
	Author: Bharadwaj Tanikella
"""
def revise(assignment, csp, var1, var2, constraint):
	inferences = set([])
	"""Question 5"""
	"""YOUR CODE HERE"""
	#if(consistent(assignment,csp,var1,constraint)) Consistency checker
	if(not constraint.affects(var1) and not constraint.affects(var2)):
		return inferences

	#if not satisfied remove the instances. 
	else:
		for variable2 in assignment.varDomains[var2]: #Variable 2 is iterated through to get a satisfaction value
			satisfaction= False
			for variable1 in assignment.varDomains[var1]: # checks with Variable1 to get the satisfaction meter.
				if(constraint.isSatisfied(variable1,variable2)):#Is the variable 2 satisfied with var 1
					satisfaction=True
					break # breaks from the loop to provide the var1 and var 2 satisfiability
			if satisfaction==False:
				inferences.add((var2,variable2))
		if len(inferences)== len(assignment.varDomains[var2]): #Check if all the variables are added, if they are returning None is an option. Or else FAIL: test_cases/q6/ac38disconnectedinconsistent2.test
			return None
		for var,val in inferences:
			assignment.varDomains[var].remove(val)

	return inferences


"""
	Implements the maintaining arc consistency algorithm.
	Inferences take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, and
	inferences made should be reversed before ending the fuction.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<<variable, value>>
		the inferences made in this call or None if inconsistent assignment
	Author: Bharadwaj Tanikella
"""
def maintainArcConsistency(assignment, csp, var, value):
	inferences = set([])
	"""Hint: implement revise first and use it as a helper function"""
	"""Question 5"""
	"""YOUR CODE HERE"""
	queue = deque()
	#adding the arc to the queue to check the consistency. 
	#Very similar algorithm to AC-3 but the only changes is that it provides a var which can be attributed. 
	for const in csp.binaryConstraints:
		if(const.affects(var)):
			queue.append((var, const.otherVariable(var), const)) 

	while(len(queue)!=0):
		variable,nextVar,constraint = queue.pop()
		value = revise(assignment,csp,variable,nextVar,constraint)
		if(value!=None):
			if(len(value) == 0):
				continue
			else:
				for const in csp.binaryConstraints:
					if (const.affects(nextVar)):
						queue.append((nextVar,const.otherVariable(nextVar),const))
				inferences = inferences | value #appends the value to the existing set of inferences, Union Operation
		else:
			for inference in inferences:
				var= inference[0]
				val= inference[1]
				assignment.varDomains[var].add(val)
			return None
	return inferences


"""
	AC3 algorithm for constraint propogation. Used as a preprocessing step to reduce the problem
	before running recursive backtracking.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		Assignment
		the updated assignment after inferences are made or None if an inconsistent assignment
	Author: Bharadwaj Tanikella
"""
def AC3(assignment, csp):
	inferences = set([])
	"""Hint: implement revise first and use it as a helper function"""
	"""Question 6"""
	"""YOUR CODE HERE"""
	# function ac3 (X, D, R1, R2)
 # // Initial domains are made consistent with unary constraints.
 #     for each x in X
 #         D(x) := { x in D(x) | R1(x) }   
 #     // 'worklist' contains all arcs we wish to prove consistent or not.
 #     worklist := { (x, y) | there exists a relation R2(x, y) or a relation R2(y, x) }
 
 #     do
 #         select any arc (x, y) from worklist
 #         worklist := worklist - (x, y)
 #         if arc-reduce (x, y) 
 #             if D(x) is empty
 #                 return failure
 #             else
 #                 worklist := worklist + { (z, x) | z != y and there exists a relation R2(x, z) or a relation R2(z, x) }
 #     while worklist not empty

	queue = deque()
	#adding the arc to the queue to check the consistency. 
	for const in csp.binaryConstraints:
		for var in csp.varDomains: #All the variables are iterated. csp.varDomains.keys() is not required. 
			if(const.affects(var)):
				queue.append((var, const.otherVariable(var), const)) 

	while(len(queue)!=0):
		variable,nextVar,constraint = queue.pop()
		value = revise(assignment,csp,variable,nextVar,constraint)
		if(value!=None):
			if(len(value) == 0):
				continue
			else:
				for const in csp.binaryConstraints:
					if (const.affects(nextVar)):
						queue.append((nextVar,const.otherVariable(nextVar),const))
				inferences = inferences | value #adds the value to the existing set of inference and increases it.
		else:
			for inference in inferences:
				var= inference[0]
				val= inference[1]
				assignment.varDomains[var].add(val)
			return None
	return assignment


"""
	Solves a binary constraint satisfaction problem.

	Args:
		csp (ConstraintSatisfactionProblem): a CSP to be solved
		orderValuesMethod (function): a function to decide the next value to try
		selectVariableMethod (function): a function to decide which variable to assign next
		inferenceMethod (function): a function to specify what type of inferences to use
		useAC3 (boolean): specifies whether to use the AC3 preprocessing step or not
	Returns:
		dictionary<string, value>
		A map from variables to their assigned values. None if no solution exists.
"""
def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic, selectVariableMethod=minimumRemainingValuesHeuristic, inferenceMethod=forwardChecking, useAC3=True):
	assignment = Assignment(csp)

	assignment = eliminateUnaryConstraints(assignment, csp)
	if assignment == None:
		return assignment

	if useAC3:
		assignment = AC3(assignment, csp)
		if assignment == None:
			return assignment

	assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
	if assignment == None:
		return assignment

	return assignment.extractSolution()