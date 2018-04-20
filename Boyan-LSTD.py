import numpy as np
import math
import matplotlib.pyplot as plt


class BoyansChain:
  p=np.array([
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0]
  ])
  phi=np.array([
   [0, 0, 0, 1],
   [0, 0, 0.25, 0.75],
   [0, 0, 0.5, 0.5],
   [0, 0, 0.75, 0.25],
   [0, 0, 1, 0],
   [0, 0.25, 0.75, 0],
   [0, 0.5, 0.5, 0],
   [0, 0.75, 0.25, 0],
   [0, 1, 0, 0],
   [0.25, 0.75, 0, 0],
   [0.5, 0.5, 0, 0],
   [0.75, 0.25, 0, 0],
   [1, 0, 0, 0]
  ])
  r=np.array([
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [-3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, -3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, -3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, -3, -3, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, -3, -3, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, -3, -3, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0]
  ])
  start=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
  end=[0]
  trueValues=np.array([-24,-16,-8, 0])

class LSTDLambda:
	def __init__ (self, problem, epsilon=0.1,gamma=0.9,lamb=0):
	    """The parameter problem passed to this function must be one of the classes
	    presented above"""
	    self.epsilon=epsilon
	    self.problem=problem
	    self.phi=problem.phi
	    self.d=self.phi.shape[0]
	    self.k=self.phi.shape[1]
	    self.identity=np.identity(self.k)
	    self.gamma=gamma
	    self.lamb=lamb
	    self.P=self.problem.p
	    self.R=self.problem.r
	    self.initialize()
	  
	def initialize(self):
	    """Initialize the matrix and vectors used for the algorithm"""
	    self.A=np.zeros([self.k,self.k])
	    self.b=np.zeros(self.k)
	    self.C=(self.epsilon)*self.identity
	    self.theta = np.zeros(self.k)
	    self.t=0
	    self.episodeLength=0
	    
	def newEpisode(self):
	    """Restores the values of some variables after finishing an episode"""
	    self.s=getInitialState(self.problem)
	    self.zt=self.phi[self.s]

	def step(self,updateMethod, optimal=False):
	    """Chooses and takes an action on the experiment, then computes the values
	    for the parameters of the algorithm. The updateMethod parameter must be 
	    one of the off-line or recursive methods of this same class."""
	    action = 0
	    
	    #According to the optimal parameter, chooses the next action. Either at 
	    #random from the transition probabilities of the experiment or by taking 
	    #the state with greater value according to te current approximated function
	    if optimal:
	      action = chooseActionOptimal(self.s,self.getStateValues(),self.P[self.s])
	    else:  
	      action = chooseAction(self.s, self.P[self.s])
	      
	    #Calculates the reward for the action and the target state 
	    r,s1=performAction(self.s,action,self.R)
	    #Executes the updating metho provided as parameter for this function.
	    updateMethod(r,s1)
	    
	    #Increases the time step
	    self.t+=1
	  
	def updateOffline(self,r,s1):
	    """Off-line approach update for the LSTD(λ) algorithm as described in 
	     Boyan's paper.
	     
	     If this method is used the updateCoeff method of this class must be used
	     later to update the weights"""
	    
	    #Calculates x(ϕ(s_t) - ϕ(s_t+1))'
	    add=np.outer(self.zt,np.transpose(self.phi[self.s]-self.phi[s1]))
	    #Updates A = A + x(ϕ(s_t) - ϕ(s_t+1))'
	    self.A+=add
	    #Updates b = b + xR
	    self.b+=self.zt*r
	    #Instead of using x as the only parameter vector for the updates we add the
	    #lambda approach by using a trace as the X in all the update steps.
	    # X = λ*x + ϕ(s_t)
	    self.zt=self.gamma*self.lamb*self.zt + self.phi[s1]
	    #Updates the current state
	    self.s=s1
	    self.episodeLength+=1

	def updateOffline_V2(self,r,s1):
	    """Off-line approach update for the LSTD(λ) algorithm as described in 
	     Boyan's paper.
	     
	     If this method is used the updateCoeff method of this class must be used
	     later to update the weights"""
	    
	    #Calculates x(ϕ(s_t) - ϕ(s_t+1))'
	    add=np.outer(self.zt,np.transpose(self.phi[self.s]-self.gamma*self.phi[s1]))
	    #Updates A = A + x(ϕ(s_t) - ϕ(s_t+1))'
	    if self.episodeLength==0:
	    	self.A+=add
	    	#Updates b = b + xR
	    	self.b+=self.zt*r
	    else:
	    	#print(float(1/(self.episodeLength+1)))
	    	self.A=((self.episodeLength)*self.A+add)*float(1/(self.episodeLength+1))
	    	self.b=((self.episodeLength)*self.b+self.zt*r)*float(1/(self.episodeLength+1))
	    	#print(self.A)
	    	#print(self.b)
	    #Instead of using x as the only parameter vector for the updates we add the
	    #lambda approach by using a trace as the X in all the update steps.
	    # X = λ*x + ϕ(s_t)
	    self.zt=self.gamma*self.lamb*self.zt + self.phi[s1]
	    #Updates the current state
	    self.s=s1
	    self.episodeLength+=1
	    
	def updateRecursive(self,r,s1):
	    """Recursive approach for the LSTD(λ) algorithm. If used do NOT call the 
	    funtion updateCoeff during the process"""
	    #Gets ϕ(s_t)
	    phi_t = self.phi[self.s]
	    #Gets ϕ(s_t+1)
	    phi_t1 = self.phi[s1]
	    
	    #Some variations were tried for the recursive algorithm.
	    
	    """Boyan's Variation 
	    Based on the off-line method proposed by Boyan we changed the update 
	    step:
	    
	    A = A + x(ϕ(s_t) - ϕ(s_t+1))'
	    
	    by calculating instead A^-1 using the Sherman-Morrison formula and after 
	    updating the parameters we also updated the coeficients."""
	    """
	    #Calculates ϕ(s_t) - γϕ(s_t+1)
	    v = phi_t - self.gamma * phi_t1
	    #Uses Sherman-Morrison's formula to calculate A^-1=(A + xv')^-1
	    self.C = ShermanMorrison(self.C, self.zt, v)
	    #Updates b = b + xR
	    self.b += self.zt*r
	    #Instead of using  ϕ_t as the only parameter vector for the updates we add the
	    #lambda approach by using a trace as the X in all the update steps.
	    # X = λ*X + ϕ(s_t)
	    self.zt = self.lamb * self.zt + self.phi[s1]
	    #Updates the weights by calculating θ = A^-1 * b 
	    self.theta = np.inner(self.C, self.b)
	    """
	    
	    """Bradke & Barto's Approach
	    Based on the RLS TD method proposed by Bradke & Barto we used the formulas:
	    
	    e_t = R_t - (ϕ_t - γϕ_t+1)'θ_t-1
	    C_t = C_t-1 - ([C_t-1*x*(ϕ_t - γϕ_t+1)'C_t-1] / [1 + (ϕ_t - γϕ_t+1)'C_t-1*x])
	    θ_t = θ_t-1 + (C_t-1 / [1 + (ϕ_t - γϕ_t+1)'C_t-1*x])e_t*x
	    
	    As this formula is the TD(0) learning rule we added the lambda parameter by
	    using X instead of only  ϕ_t to turn it into a discounted trace:
	    X = λ*X + ϕ_t
	    """
	    #Calculates Cϕ = C_t-1 * x
	    CPhi = np.dot(self.C, self.zt)
	    #Calculates v = (ϕ(s_t) - γϕ(s_t+1))'
	    v = np.transpose(phi_t - self.gamma*phi_t1)
	    #Updates e_t = R_t - vθ_t-1
	    et = r - np.dot(v, self.theta)
	    #Calculates d = C_t-1/(1 + vCϕ). The division portion of the  updating rule.
	    d = self.C / (1 + np.dot(v,CPhi))
	    #Calculates C_t = C_t-1 - (C_t-1*x*(ϕ_t - γϕ_t+1)')d
	    ct_1 = self.C - np.dot(np.dot(self.C,np.outer(self.zt,v)),d)
	    #Updates θ_t = θ_t-1 + d*e_t*x
	    self.theta += np.dot(d,et*self.zt)
	    #Updates C_t-1 t C_t
	    self.C=ct_1
	    #Instead of using  ϕ_t as the only parameter vector for the updates we add the
	    #lambda approach by using a trace as the X in all the update steps.
	    # X = λ*X + ϕ(s_t)
	    self.zt = self.zt*self.lamb + phi_t1
	    
	    
	    """Sutton & Barto's Approach
	    Based on the LSTD algorithm proposed by Sutton & Barto we used the 
	    algrithm:
	    C = A^-1
	    
	    v = C_t'(ϕ_t - γϕ(t+1))
	    C_t = C_t-1 - (C_t-1 * X)v' / (1 + v'X)
	    b = b + RX
	    θ = C_t * b
	    
	    As this formula is an LSTD(0) learning rule we added the lambda parameter by
	    using X instead of only  ϕ_t to turn it into a discounted trace:
	    X = λ*X + ϕ_t"""
	    """
	    # Calculates v = C_t'(ϕ_t - γϕ(t+1))
	    v = np.dot(np.transpose(self.C),(phi_t - self.gamma*phi_t1))
	    # Calculates C_t = C_t-1 - (C_t-1 * x)v' / (1 + v'x)
	    self.C = self.C - np.dot(np.dot(self.C,self.zt),np.transpose(v))/ (1+ np.dot(np.transpose(v),self.zt))
	    #Updates b = b + xR
	    self.b += r*self.zt
	    #Updates θ = C_t * b
	    self.theta = np.inner(self.C,self.b)
	    #Instead of using  ϕ_t as the only parameter vector for the updates we add the
	    #lambda approach by using a trace as the X in all the update steps.
	    # X = λ*X + ϕ(s_t)
	    self.zt = self.zt*self.lamb + phi_t1
	    """
	    #Updates the current state
	    self.s=s1
	      
	def updateCoeff(self):
	    """Updates the Coefficients by calculating the inner product A^-1*b"""
	    self.theta=np.inner(np.linalg.pinv(self.A),self.b)
	  
	def getStateValues(self):
	    """Calculates the state values by V(x)= θϕ(s)"""
	    values=[np.dot(self.theta,phi_x) for phi_x in self.phi]
	    return values
	

def RMS(data,true):
	"""Function to calculate the RMS error of the data"""
	mse=[((true - value)**2).mean(axis=None) for value in data]
	return np.array([math.sqrt(value) for value in mse])

def getInitialState(problem):
	"""Selects an initial state from the initialprobabilities"""
	#Makes sure the probabilities sum 1
	st_p = problem.start/sum(problem.start)
	#Randomly picks the starting state
	state = np.random.choice(len(st_p),p=st_p)
	return state

def chooseActionOptimal(state, values, transition):
	"""Selects an action to execute on a state based on the transition 
	probabilities and the state-values. This function will always pick the state
	with highest value where a transition exists. Greedy policy"""

	#Checks which transitions do exist
	index=np.argwhere(transition)
	#Makes the state-value of the unreachable states really low so they can't be 
	#selected
	value = [(-np.inf if ind not in index else x) for ind,x in enumerate(values)]
	#Searches for the highest state-value.
	top = np.amax(value)
	#If more than one state have the same state-value picks randomly one of them.
	optimals = np.argwhere(value == top).flatten()
	action = np.random.choice(optimals)

	return action

def chooseAction(state, transition_prob, optimal=False):
	"""Selects an action to execute on a state based on the transition 
	probabilities"""

	#Makes sure the transition probabilities sum 1
	prob=transition_prob/sum(transition_prob)
	#Picks a random state based on their transition probability 
	action=np.random.choice(len(prob),p=prob)

	return action

def ShermanMorrison(A_1,x,v):
	"""Calculates the value of (A +xv')^-1 using the  Sherman-Morrison's formula
	given (A_t-1)^-1, x and v"""

	#Calculates the new A_t value with the formula
	#(A_t)^-1 = (A_t-1)^-1 - ([(A_t-1)^-1*x*v'(A_t-1)^-1] / [1 + v'(A_t-1)^-1*x])
	A_t= A_1 - np.dot(np.dot(A_1,np.outer(x,np.transpose(v))),A_1) / (1 + np.dot(np.transpose(v),np.dot(A_1,x)))

	return A_t

def performAction(state, action, reward):
	"""Calculates the inmediate reward and target of state of performing 
	certain action being at some state"""
	return reward[state,action],action

episodes=10
trials=10000
np.random.seed(0)
problem=BoyansChain

lambdaValues=[0,0.2,0.4,0.6,0.8,1.0]
plotValues=[]
plotValues1=[]

for lam in lambdaValues:
  lstd=LSTDLambda(problem,lamb=lam,gamma=0.9)
  lambdaPlot=[]
  
  for trial in range(0,trials):
    trialValues=[]
    for episode in range(0,episodes):
      lstd.newEpisode()
      lstd.episodeLength=0
      while lstd.s not in problem.end:
      	#print("Timestep is: "+str(lstd.episodeLength))
      	lstd.step(lstd.updateOffline_V2,False)
      lstd.updateCoeff()
      #print(lstd.theta)
      trialValues.append(lstd.theta)

      
    lambdaPlot.append(RMS(trialValues, problem.trueValues).mean(axis=None))
  
  plotValues.append(lambdaPlot)

fig, ax= plt.subplots()
d=np.asarray(plotValues)
for i,line in enumerate(d):
  plt.plot(range(0,len(line)),line, label="lambda = " + str(lambdaValues[i]))
  
plt.axis([0,10000,0,0.2])

plt.xlabel("Trajectory number")
plt.ylabel("Error")
plt.title("Off-line")
plt.legend()
plt.axis('auto')
plt.show()
