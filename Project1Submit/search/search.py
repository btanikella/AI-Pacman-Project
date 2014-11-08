# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# CS 3600, Gatech
# editor: Bharadwaj Tanikella 

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print problem.getStartState()
    #print problem.isGoalState(problem.getStartState())
    stackDFS= util.Stack() #obtain the stack to keep track of the nodes.
    visitedLibrary = []
    resultingArray = []
    corners={}
    currentState= problem.getStartState()
    corners[currentState]=(("0",("0","0")))

    stackDFS.push(currentState)

    while(stackDFS.isEmpty()!=True):
        currentState= stackDFS.pop()
        visitedLibrary.append(currentState)
        if(problem.isGoalState(currentState)==True):
            while(currentState!=problem.getStartState()):
                resultingArray.append(corners.get(currentState)[0])
                currentState= corners.get(currentState)[1]
            answer=[]
            for i in range(resultingArray.__len__(),0,-1):
                answer.append(resultingArray[i-1])
            return answer


        for l in problem.getSuccessors(currentState):
            value = l[0]
            next = l[1]
            if(visitedLibrary.__contains__(value)==False):
                stackDFS.push(value)
                corners[value]=(next,currentState)
    return resultingArray


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    queueBFS = util.Queue() #Obtain the datastructure to provide and use to keep track of the nodes.
    
    visitedLibrary = []
    resultingArray = []
    corners = {}
    currentState = problem.getStartState()
    corners[currentState] = (("0",("0","0"))) #Direction, (Coordinates) of the State. 

    queueBFS.push(currentState)

    while((queueBFS.isEmpty()==False)):      

        currentState = queueBFS.pop()
        visitedLibrary.append(currentState)


        if(problem.isGoalState(currentState)==True):
            while(currentState != problem.getStartState()):
                resultingArray.append(corners.get(currentState)[0])
                currentState = corners.get(currentState)[1]
            answer = []
            for i in range(resultingArray.__len__(),0,-1):
                answer.append(resultingArray[i-1])
            return answer

        for l in problem.getSuccessors(currentState):
            value = l[0]
            next  = l[1]
            if((corners.get(value)==None) and ((visitedLibrary.__contains__(value))==False)):
                queueBFS.push(value);
                corners[value]=(next,currentState) 
    return resultingArray

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
  #   procedure UniformCostSearch(Graph, root, goal)
  # node := root, cost = 0
  # frontier := priority queue containing node only
  # explored := empty set
  # do
  #   if frontier is empty
  #     return failure
  #   node := frontier.pop()
  #   if node is goal
  #     return solution
  #   explored.add(node)
  #   for each of node's neighbors n
  #     if n is not in explored
  #       if n is not in frontier
  #         frontier.add(n)
  #       else if n is in frontier with higher cost
  #         replace existing node with n
  
    ucsPQ = util.PriorityQueue()
    
    visitedLibrary = []
    resultingArray = []
    cornersDict = {}
    
   
    currentState = problem.getStartState()


    cornersDict[currentState] = (("0",("0","0"),0))
    ucsPQ.push(problem.getStartState(),0)
    


    while((ucsPQ.isEmpty()==False)):      
        currentState = ucsPQ.pop()
        visitedLibrary.append(currentState)
        if(problem.isGoalState(currentState)==True):
            while(currentState != problem.getStartState()):
                resultingArray.append(cornersDict.get(currentState)[0])
                currentState = cornersDict.get(currentState)[1]

            #print reverse    
            answer = []
            for i in range(resultingArray.__len__(),0,-1):
                answer.append(resultingArray[i-1])
            return answer
        
        for l in problem.getSuccessors(currentState):
            value = l[0]
            next  = l[1]
            parent =l[2]
            if((cornersDict.get(value)==None) or(cornersDict.get(l[0])[2]) > parent+cornersDict.get(currentState)[2])and ((visitedLibrary.__contains__(value))==False): 
                ucsPQ.push(value,parent+cornersDict.get(currentState)[2]);
                cornersDict[value]=(next,currentState,(cornersDict.get(currentState)[2]+parent)) 
                
                    
    return resultingArray


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
#     The psuedocode for aStarSearch
#     function A*(start,goal)
#     closedset := the empty set    // The set of nodes already evaluated.
#     openset := {start}    // The set of tentative nodes to be evaluated, initially containing the start node
#     came_from := the empty map    // The map of navigated nodes.
 
#     g_score[start] := 0    // Cost from start along best known path.
#     // Estimated total cost from start to goal through y.
#     f_score[start] := g_score[start] + heuristic_cost_estimate(start, goal)
 
#     while openset is not empty
#         current := the node in openset having the lowest f_score[] value
#         if current = goal
#             return reconstruct_path(came_from, goal)
 
#         remove current from openset
#         add current to closedset
#         for each neighbor in neighbor_nodes(current)
#             if neighbor in closedset
#                 continue
#             tentative_g_score := g_score[current] + dist_between(current,neighbor)
 
#             if neighbor not in openset or tentative_g_score < g_score[neighbor] 
#                 came_from[neighbor] := current
#                 g_score[neighbor] := tentative_g_score
#                 f_score[neighbor] := g_score[neighbor] + heuristic_cost_estimate(neighbor, goal)
#                 if neighbor not in openset
#                     add neighbor to openset
 
#     return failure
 
# function reconstruct_path(came_from, current_node)
#     if current_node in came_from
#         p := reconstruct_path(came_from, came_from[current_node])
#         return (p + current_node)
#     else
#         return current_node
    aStarPQ = util.PriorityQueue()
   
    visitedLibrary = []
    resultingArray = []
    cornersDict = {}
    currentState = problem.getStartState()
   
    cornersDict[currentState] = (("0",("0","0"),heuristic(currentState, problem)))
    aStarPQ.push(currentState,0)

    while((aStarPQ.isEmpty()==False)):      
        currentState = aStarPQ.pop()
        if(visitedLibrary.__contains__(currentState)==True):
            continue
        visitedLibrary.append(currentState)
        if(problem.isGoalState(currentState)==True):
            while(currentState != problem.getStartState()):
                resultingArray.append(cornersDict.get(currentState)[0])
                currentState = cornersDict.get(currentState)[1]
           
            answer = []
            for i in range(resultingArray.__len__(),0,-1):
                answer.append(resultingArray[i-1])
            return answer

        for l in problem.getSuccessors(currentState):
            value = l[0]
            next  = l[1]
            parent =l[2]
            if((cornersDict.get(value)==None) or(cornersDict.get(value)[2]) > parent+cornersDict.get(currentState)[2]) and ((visitedLibrary.__contains__(value))==False): 
                aStarPQ.push(value,(parent+cornersDict.get(currentState)[2] + heuristic(value, problem)));
                cornersDict[value]=(next,currentState,(cornersDict.get(currentState)[2]+parent)) 
                    
    return resultingArray



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
