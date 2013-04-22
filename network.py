#!/usr/bin/python2
import random
import math

class network(object):
	"""Create and initialize the network neurons and weights"""
	def __init__(self, inputs, hidden_layers, hidden_neurons, outputs):
		self.inputs 		= inputs+1 			#accaunting for the bias
		self.hidden_layers 	= hidden_layers
		self.hidden_neurons = hidden_neurons+1 	#accaunting for the bias
		self.outputs 		= outputs

		#Array with variable amount of inputs
		self.i_n	= [0 for h in xrange(self.inputs)]
		#Array with variable hidden layers and neurons
		self.h_n 	= [[0 for h in xrange(self.hidden_neurons)] for h in xrange(self.hidden_layers)]
		#Ouput array for the final output neurons
		self.o_n	= [0 for h in xrange(self.outputs)]
		
		#initialising bias
		self.i_n[self.inputs-1] = -1
		if(self.hidden_layers > 1):
			for h in xrange(self.hidden_layers):
				self.h_n[h][self.hidden_neurons-1] = -1
		else:
			self.h_n[0][self.hidden_neurons-1] = -1

		#Weights get initialized with a random number between -0.5 and 0.5
		random.seed()
		#Weights from input neurons to the hidden layer neurons
		self.w_ih	= [[random.uniform(-0.5,0.5) for h in xrange(self.inputs)]for h in xrange(self.hidden_neurons-1)]
		#Weights from hidden layer to hidden layers
		if(hidden_layers > 1):
			self.w_hh	= [[[random.uniform(-0.5,0.5) for h in xrange(hidden_neurons)] for h in xrange(hidden_neurons)] for h in xrange(hidden_layers-1)]
		#Weights from last hidden neurons to the output neurons
		self.w_ho 	= [[random.uniform(-0.5,0.5) for h in xrange(hidden_neurons)] for h in xrange(outputs)]		
		print self.w_ih,"\n--------"
		print len(self.w_ih)
		print self.w_hh
		print len(self.w_hh)

	"""Calculate the output of every neuron using the sigmoid function
	Could be extended with a choide in step functions"""
	def activate(self):
		#From input to the first hidden layer
		for h in range(0,self.hidden_neurons-1):
			self.h_n[0][h]	= 0	#zeroing the neuron output
			for j in range(0,self.inputs):
				self.h_n[0][h] += self.i_n[j]*self.w_ih[h][j]
#				print h,j,self.h_n[0][h],"|",self.i_n[j],"*",self.w_ih[j][h]
			self.h_n[0][h] = self.sigmoid(self.h_n[0][h])
		print self.h_n

		#From hidden to hidden of there are more than 1 hidden layer
		if(self.hidden_layers > 1):
			for h in range(1,self.hidden_layers):
				for j in range(0,self.hidden_neurons):
#					print j,self.hidden_neurons
					self.h_n[h][j] = 0
					for k in range(0,self.hidden_neurons-1):
						print h,j,k,self.h_n[h][j], self.w_hh[h][k][j]
						self.h_n[h][j] += self.h_n[h-1][k]*self.w_hh[h][k][j]
					print j,"cycle complete"
					self.h_n[h][j] = self.sigmoid(self.h_n[h][j])
		print self.h_n





	def sigmoid(self,x):
		return 1/(1+math.exp(x))

	def step(self, x):	
		if (x >= 0):
			return 1
		elif(x < 0):
			return 0	
		print "step"
	def properties(self):
		print "Inputs:",self.inputs,"\nHidden layers:",self.hidden_layers,"\nHidden neurons:",self.hidden_neurons,"\nOutputs:",self.outputs

net = network(2,2,5,2)
net.properties()
net.activate()