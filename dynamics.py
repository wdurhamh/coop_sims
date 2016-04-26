"""
TODO:
	- Some sort of visualization
	- Put hard bounds (0<=beta<=1, etc...)
"""


import numpy as np
import pandas as pd
from scipy.integrate import odeint

class Dynamcis:

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
		self.sim_data['Pi1'] = self.sim_data["x1"]*(self.A[0,0]*self.sim_data["x1"] + self.A[0,1]*self.sim_data["x2"] + self.B[0,0])
		self.sim_data['Pi2'] = self.sim_data["x2"]*(self.A[1,0]*self.sim_data["x1"] + self.A[1,1]*self.sim_data["x2"] + self.B[1,0])
		self.sim_data['p1sur'] = (self.sim_data['x1']*self.A[0,1]*self.sim_data['dx2'])
		self.sim_data['p2sur'] = (self.sim_data['x2']*self.A[1,0]*self.sim_data['dx1'])
		self.sim_data['Pi1NC']  = self.sim_data['x1']*(self.A[0,0]*self.sim_data['x1'] + self.A[0,1]*self.sim_data['barx2'] + self.B[0,0])
		self.sim_data['Pi2NC']  = self.sim_data['x2']*(self.A[1,1]*self.sim_data['x2'] + self.A[1,0]*self.sim_data['barx1'] + self.B[1,0])
		self.sim_data['V1'] = self.sim_data['Pi1NC'] + (1 - self.sim_data['beta1'])*self.sim_data['p1sur'] + self.sim_data['beta2']*self.sim_data['p2sur']
		self.sim_data['V2'] = self.sim_data['Pi2NC'] + (1 - self.sim_data['beta2'])*self.sim_data['p2sur'] + self.sim_data['beta1']*self.sim_data['p1sur']
		self.sim_data['Social Welfare'] = self.sim_data["Pi1"] + self.sim_data["Pi2"]
		self.sim_data = self.sim_data[['barx1', 'barx2', 'dx1', 'dx2', 'x1', 'x2', 'beta1','beta2', 'V1', 'V2', 'Social Welfare']]


class NC(Dynamcis):

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

class FC(Dynamcis):

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

class SidePayment(Dynamcis):

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
		beta = x[3]
		dx1 = x[4]
		alpha = x[5]
		x1 = barx1 + dx1
		x2 = barx2 + dx2
		

		dx1_dot = alpha*A[1,0]*x2 + (2*A[0,0]*x1 + A[0,1]*x2 + B[0] ) + (1-beta)*(dx2*A[0,1])
		dx1_opt = ( alpha*A[1,0]*x2 + (2*A[0,0]*barx1 + A[0,1]*x2 + B[0]) )/(-2*A[0,0])
		alpha_dot = A[1,0]*A[1,0]*x2/(-2*A[0,0]) - 1*(x2*dx1_opt*A[1,0]) + ((A[1,0]*x2)**2)*(1-alpha)/(-2*A[0,0]) 
	    
		dx2_dot = beta*A[0,1]*x1 + (2*A[1,1]*(x2) + A[1,0]*x1 + B[1] ) + (1-alpha)*(dx1*A[1,0])
		dx2_opt = ( x[3]*A[0,1]*x[0] + (2*A[1,1]*x[1] + A[1,0]*x[0] + B[1]) )/(-2*A[1,1])
		#dx2_opt = 1
		beta_dot = A[0,1]*A[0,1]*x1/(-2*A[1,1]) - 1*(x1*dx2_opt*A[0,1]) + ((A[0,1]*x1)**2)*(1-beta)/(-2*A[1,1])
		x_dot = x_dot.reshape(2,).tolist()[0]
	    
	    #multiplicative constant
		c = 1.0
		x_dot.append(c*dx2_dot[0,0])
		x_dot.append(c*beta_dot[0,0])
		x_dot.append(c*dx1_dot[0,0])
		x_dot.append(c*alpha_dot[0,0])
	    
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


