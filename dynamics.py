"""
Cooperation simulation in two-firm Bertrand market with linear demand
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy import linalg

class Dynamics:

	"""
	Initialize the market with A and B matricies.
	For now, we are assuming 2 player markets
	"""
	def __init__(self, A, B, n=2):
		self.A = A
		self.B = B
		self.n = n
		self.set_initial_conditions()
		self.t = np.linspace(0.0, 30, 10000)

	""" 
	Expexting a simple list, not an numpy array. It's just easier that way.
	Also, I think it's more lightweight 
	"""
	def set_initial_conditions(self, x=[0,0]):
		self.init_cond = x

	"""
	Sets the interval over which the simulation is run
	"""
	def set_t(self, t):
		self.t = t

	"""
	A place holder to be overwritten in sub classes
	"""
	def dynamics(self,x,t):
		return [0,0]

	"""
	Method stub to be overwritten by sub classes
	"""
	def calc_additional_vars(self):
		return 'Implemented by subclasses'

	"""
	Run the simulation from the initial condition (default is all 0's)
	over the interval (default is 10000 intervals between 0 and 30)
	"""
	def run_simulation(self):
		sim_data = odeint(self.dynamics, self.init_cond, self.t)
		self.sim_data = pd.DataFrame(sim_data)
		self.name_cols()
		self.calc_additional_vars()
		return self.sim_data

	"""
	Calculates system variables from the three fundmental variabels, barx, dx, and beta
	Not overwritten in any class
	"""
	def calc_prices_profits_and_revenues(self):
		self.sim_data['x1'] = self.sim_data['barx1'] + self.sim_data['dx1']
		self.sim_data['x2'] = self.sim_data['barx2'] + self.sim_data['dx2']
		self.sim_data['Pi1'],self.sim_data['Pi2'] = self.get_profits(self.sim_data["x1"],self.sim_data["x2"])
		self.sim_data['p1sur'] = (self.sim_data['x1']*self.A[0,1]*self.sim_data['dx2'])
		self.sim_data['p2sur'] = (self.sim_data['x2']*self.A[1,0]*self.sim_data['dx1'])
		self.sim_data['Pi1NC']  = self.sim_data['x1']*(self.A[0,0]*self.sim_data['x1'] + self.A[0,1]*self.sim_data['barx2'] + self.B[0,0])
		self.sim_data['Pi2NC']  = self.sim_data['x2']*(self.A[1,1]*self.sim_data['x2'] + self.A[1,0]*self.sim_data['barx1'] + self.B[1,0])
		self.sim_data['V1'] = self.sim_data['Pi1NC'] + (1 - self.sim_data['beta1'])*self.sim_data['p1sur'] + self.sim_data['beta2']*self.sim_data['p2sur']
		self.sim_data['V2'] = self.sim_data['Pi2NC'] + (1 - self.sim_data['beta2'])*self.sim_data['p2sur'] + self.sim_data['beta1']*self.sim_data['p1sur']
		self.sim_data['Producer Welfare'] = self.sim_data["Pi1"] + self.sim_data["Pi2"]
		self.sim_data = self.sim_data[['barx1', 'barx2', 'dx1', 'dx2', 'x1', 'x2', 'beta1','beta2', 'V1', 'V2', 'Producer Welfare']]
	"""
	Generates data for contour plot
	"""	
	def gen_contour_graph_data(self,xmin=0,xmax=10,ymin=0,ymax=10):
		if self.n!=2: return "gen_contour_graph only works for two player systems"
		xi, yi = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
		xi,yi=np.meshgrid(xi,yi)
		pw=self.get_producer_welfare(xi,yi)
		return xi,yi,pw

	"""
	Given prices, calc profits
	"""
	def get_profits(self,x1,x2):
		pi1=x1*(self.A[0,0]*x1 + self.A[0,1]*x2 + self.B[0,0])
		pi2=x2*(self.A[1,1]*x2 + self.A[1,0]*x1 + self.B[1,0])
		return pi1,pi2
	"""
	Calculate producer welfare at a given pricing policy
	"""
	def get_producer_welfare(self,x1,x2):
		pi1,pi2=self.get_profits(x1,x2)
		return pi1+pi2

	"""
	Get prices at the non-cooperative equilibrium
	"""	
	def get_NCE_prices(self):
		_A = self.A + np.array([[self.A[0,0], 0],[0, self.A[1,1]]]).reshape(2,2)
		policy=linalg.solve(_A,-1*self.B)
		return policy[0,0],policy[1,0]
	"""
	Get prices at the perfectly cooperative or collusive outcome
	"""
	def get_PCE_prices(self):
		_A = self.A + np.array([[self.A[0,0], self.A[1,0]],[self.A[0,1], self.A[1,1]]]).reshape(2,2)
		policy=linalg.solve(_A,-1*self.B)
		return policy[0,0],policy[1,0]

class NC(Dynamics):

	def dynamics(self, x,t):
		x = x.reshape(2,1)
		_A = self.A + np.array([[self.A[0,0], 0],[0, self.A[1,1]]]).reshape(2,2)
		xdot = np.array([0,0]).reshape(2,1)
		xdot = np.dot(_A,x) + self.B
		return xdot.reshape(2,).tolist()[0]

	def name_cols(self):
		self.sim_data.columns=["barx1","barx2"]

	def calc_additional_vars(self):
		self.sim_data['dx1'] = 0
		self.sim_data['dx2'] = 0
		self.sim_data['beta1'] = 0
		self.sim_data['beta2'] = 0
		self.calc_prices_profits_and_revenues()

class FC(Dynamics):

	def dynamics(self,x,t):
		x = x.reshape(2,1)
		_A = self.A + np.array([[self.A[0,0], self.A[1,0]],[self.A[0,1], self.A[1,1]]]).reshape(2,2)
		xdot = np.array([0,0]).reshape(2,1)
		xdot = np.dot(_A,x) + self.B
		return xdot.reshape(2,).tolist()[0]

	def name_cols(self):
		self.sim_data.columns=["barx1","barx2"]

	def calc_additional_vars(self):
		self.sim_data['dx1'] = 0
		self.sim_data['dx2'] = 0
		self.sim_data['beta1'] = 0
		self.sim_data['beta2'] = 0
		self.calc_prices_profits_and_revenues()

class SidePayment(Dynamics):

	def set_initial_conditions(self,x=[0,0,0,0,0,0]):
		self.init_cond = x

	def dynamics(self,x,t):
		x = np.matrix(x).T
		J = np.array( [[2*self.A[0,0], self.A[0,1],0,0,0,0], [self.A[1,0], 2*self.A[1,1],0,0,0,0]] ).reshape(2,6)
		A = self.A
		B = self.B
		x_dot = J*x + B
		barx1 = x[0]
		barx2 = x[1]
		dx2 = x[2]
		beta1 = x[3]
		dx1 = x[4]
		beta2 = x[5]
		x1 = barx1 + dx1
		x2 = barx2 + dx2

		dx1_dot = beta2*A[1,0]*x2 + (2*A[0,0]*x1 + A[0,1]*barx2 + B[0] ) + (1-beta1)*(dx2*A[0,1])
		dx1_opt = ( beta2*A[1,0]*x2 + (2*A[0,0]*barx1 + A[0,1]*barx2 + B[0]) )/(-2*A[0,0])
		dx2_dot = beta1*A[0,1]*x1 + (2*A[1,1]*(x2) + A[1,0]*barx1 + B[1] ) + (1-beta2)*(dx1*A[1,0])
		dx2_opt = ( beta1*A[0,1]*x1 + (2*A[1,1]*barx2 + A[1,0]*barx1 + B[1]) )/(-2*A[1,1])
	    
	    #betaj
		beta2_dot = -1*(x2*dx1_opt*A[1,0]) + ((A[1,0]*x2)**2)*(1-beta2)/(-2*A[0,0]) #+ beta*dx2*A[1,0]*A[0,1]*x2/(-2*A[0,0])
	    #betai
		beta1_dot = -1*(x1*dx2_opt*A[0,1]) + ((A[0,1]*x1)**2)*(1-beta1)/(-2*A[1,1]) #+ alpha*dx1*A[0,1]*A[1,0]*x1/(-2*A[1,1])
	    
		x_dot = x_dot.reshape(2,).tolist()[0]
	    
	    #multiplicative constant
		c = 1.0
		x_dot.append(c*dx2_dot[0,0])
		x_dot.append(c*beta1_dot[0,0])
		x_dot.append(c*dx1_dot[0,0])
		x_dot.append(c*beta2_dot[0,0])
	    
	    #check limits (though I don't think these should be necessary)
	    #check that beta is between 0 and 1
		if x[3] >= 1 and x_dot[3] > 0:
			x_dot[3] = 0
		if x[3] <= 0 and x_dot[3] < 0:
			x_dot[3] = 0
	        
		if x[5] >= 1 and x_dot[5] > 0:
			x_dot[5] = 0
		if x[5] <= 0 and x_dot[5] < 0:
			x_dot[5] = 0
	    #make sure delta is not greater than x2
	    #perhaps here we need to think of coeficients
		if (x[1] + x[2]) <=0 and ( x_dot[2] + x_dot[1] ) < 0:
			x_dot[2] = 0
	    #print x_dot, x
		return x_dot

	def name_cols(self):
		self.sim_data.columns = ['barx1', 'barx2', 'dx2', 'beta1', 'dx1', 'beta2']

	def calc_additional_vars(self):
		self.calc_prices_profits_and_revenues()

class SidePaymentBeta(SidePayment):

	def dynamics(self,x,t):
		x = np.matrix(x).T
		J = np.array( [[2*self.A[0,0], self.A[0,1],0,0,0,0], [self.A[1,0], 2*self.A[1,1],0,0,0,0]] ).reshape(2,6)
		A = self.A
		B = self.B
		x_dot = J*x + B
		barx1 = x[0]
		barx2 = x[1]
		dx2 = x[2]
		beta1 = x[3]
		dx1 = x[4]
		beta2 = x[5]
		x1 = barx1 + dx1
		x2 = barx2 + dx2

		dx1_dot = beta2*A[1,0]*x2 + (2*A[0,0]*x1 + A[0,1]*barx2 + B[0] ) + (1-beta1)*(dx2*A[0,1])
		dx1_opt = ( beta2*A[1,0]*x2 + (2*A[0,0]*barx1 + A[0,1]*barx2 + B[0]) )/(-2*A[0,0])
		dx2_dot = beta1*A[0,1]*x1 + (2*A[1,1]*(x2) + A[1,0]*barx1 + B[1] ) + (1-beta2)*(dx1*A[1,0])
		dx2_opt = ( beta1*A[0,1]*x1 + (2*A[1,1]*barx2 + A[1,0]*barx1 + B[1]) )/(-2*A[1,1])
	    
	    #betaj
		beta2_dot = ((1-2*beta2)*x2*A[1,0]**2 + (1-beta1)*A[0,1]*A[1,0]*x2)/(-2*A[0,0]) #+ beta*dx2*A[1,0]*A[0,1]*x2/(-2*A[0,0])
	    #betai
		beta1_dot = ((1-2*beta1)*x1*A[0,1]**2 + (1-beta2)*A[1,0]*A[0,1]*x1)/(-2*A[1,1]) #+ alpha*dx1*A[0,1]*A[1,0]*x1/(-2*A[1,1])
	    
		x_dot = x_dot.reshape(2,).tolist()[0]
	    
	    #multiplicative constant
		c = 1.0
		x_dot.append(c*dx2_dot[0,0])
		x_dot.append(c*beta1_dot[0,0])
		x_dot.append(c*dx1_dot[0,0])
		x_dot.append(c*beta2_dot[0,0])
	    
	    #check limits (though I don't think these should be necessary)
	    #check that beta is between 0 and 1
		if x[3] >= 1 and x_dot[3] > 0:
			x_dot[3] = 0
		if x[3] <= 0 and x_dot[3] < 0:
			x_dot[3] = 0
	        
		if x[5] >= 1 and x_dot[5] > 0:
			x_dot[5] = 0
		if x[5] <= 0 and x_dot[5] < 0:
			x_dot[5] = 0
	    #make sure delta is not greater than x2
	    #perhaps here we need to think of coeficients
		if (x[1] + x[2]) <=0 and ( x_dot[2] + x_dot[1] ) < 0:
			x_dot[2] = 0
	    #print x_dot, x
		return x_dot
