#!/usr/bin/python2
import random
import math
import pprint
import pygame	
pp = pprint.PrettyPrinter(indent=4)

class network(object):
	def __init__(self, inputs, hidden_layers, hidden_neurons, outputs):
		"""
		Create and initialize the network neurons and weights
		"""
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
				self.h_n[h][self.hidden_neurons-1] 	= -1
		else:
			self.h_n[0][self.hidden_neurons-1] 		= -1

		#Weights get initialized with a random number between -0.5 and 0.5
		random.seed()
		#Weights from input neurons to the hidden layer neurons
		self.w_ih		= [[random.uniform(-0.5,0.5) for h in xrange(self.hidden_neurons-1)]for h in xrange(self.inputs)]
		#Weights from hidden layer to hidden layers
		if(self.hidden_layers > 1):
			self.w_hh	= [[[random.uniform(-0.5,0.5) for h in xrange(self.hidden_neurons-1)] for h in xrange(self.hidden_neurons)] for h in xrange(hidden_layers-1)]
		#Weights from last hidden neurons to the output neurons
		self.w_ho 		= [[random.uniform(-0.5,0.5) for h in xrange(self.outputs)] for h in xrange(self.hidden_neurons)]		

		#The error and error gradient variables
		#Sum of Squared Errors
		self.sse		= 0

	
	def activate(self):
		"""
		Calculate the output of every neuron using the sigmoid function
		Could be extended with a choide in step functions
		"""
		#From input to the first hidden layer
		for h in xrange(0,self.hidden_neurons-1):
			#zeroing the neuron output
			self.h_n[0][h]	= 0	
			for j in xrange(0,self.inputs):
				self.h_n[0][h] += self.i_n[j]*self.w_ih[j][h]

			self.h_n[0][h] = self.sigmoid(self.h_n[0][h])

		#From hidden to hidden of there are more than 1 hidden layer
		if(self.hidden_layers > 1):
			#For each hidden layer
			for h in xrange(0,self.hidden_layers-1):
				#For each hidden neuron input
				for j in xrange(0,self.hidden_neurons-1):
					#zeroing the neuron output
					self.h_n[h+1][j] = 0;
					#For each hidden neuron that has to be calculated
					#The bias of the next layer is always -1 and does not have to be calculated
					for k in xrange(0,self.hidden_neurons-1): 
						self.h_n[h+1][j] += self.h_n[h][j]*self.w_hh[h][k][j]
					#Apply the sigmoid function
					self.h_n[h+1][j] = self.sigmoid(self.h_n[h+1][j])

		#From last hidden layer to the output layer
		for h in xrange(0,self.outputs):
			#zeroing the neuron output
			self.o_n[h]	= 0	
			for j in xrange(0,self.hidden_neurons):
				self.o_n[h] += self.h_n[self.hidden_layers-1][j]*self.w_ho[j][h]
			self.h_n[0][h] = self.sigmoid(self.h_n[0][h])

	def backPropegate(self, desired_out, learning_rate):
		#Error gradient delta k
		delta_k 	= [0 for h in xrange(self.outputs)]
		#Error gradient delta j
		delta_j	= [[0 for h in xrange(self.hidden_neurons)] for h in xrange(self.hidden_layers)]


		#Setting the sum of squared errors to zero
		self.sse = 0
		#Calculating the error, delta k and sum of squared errors
		for h in xrange(0,self.outputs):
			error 			= desired_out[h] - self.o_n[h]
			delta_k[h] = self.o_n[h] * (1-self.o_n[h]) * error * learning_rate
			self.sse		+= error * error

		#Calculating the error gradient delta j
		for h in xrange(0,self.hidden_neurons):
			delta_j[self.hidden_layers-1][h] = 0;
			for j in xrange(0,self.outputs):
				delta_j[self.hidden_layers-1][h] += self.w_ho[h][j] * delta_k[j]
			delta_j[self.hidden_layers-1][h] = self.h_n[self.hidden_layers-1][h] * (1-self.h_n[self.hidden_layers-1][h]) * delta_j[self.hidden_layers-1][h]

		#Calculating the error gradient delta j for other hidden layers

	def activation(self, x):
		return self.sigmoid(x)

	#Hyperbolic tangent activation function
	def hypTangent(self, x):
		#a & b Chosen by Guyon, 1991
		a = 1.716
		b = 0.667
		return ((2*a)/(1+math.exp(-b*x)))-a

	def sigmoid(self, x):
		return 1/(1+math.exp(x))

	def properties(self):
		print "Inputs:",self.inputs,"\nHidden layers:",self.hidden_layers,"\nHidden neurons:",self.hidden_neurons,"\nOutputs:",self.outputs

	def showNet(self):
		"""
		Drawing the network graphically
		"""
		pygame.init() 
		distance = 80
		window = pygame.display.set_mode((((self.hidden_layers+2)*distance)+distance+(distance/2),((self.hidden_neurons+1)*distance)+20))
		#Drawing the iputs, hidden neurons and ouput neurons
		#Drawing the input rectangles
		for h in xrange(0,self.inputs):
			if self.i_n[h] < 0: 			color = (255,0,0)
			elif self.i_n[h] == 0: 			color = (255,255,255)
			else: 							color = (0,255,0)
			pygame.draw.rect(window, color, (distance,(h+1)*distance,20,20),1)

		#Drawing the hidden layer circles
		for h in xrange(0,self.hidden_layers):
			for j in xrange(0,self.hidden_neurons):
				if self.h_n[h][j] < 0: 		color = (255,0,0)
				elif self.h_n[h][j] == 0: 	color = (255,255,255)
				else: 						color = (0,255,0)
				pygame.draw.circle(window,color,(((h+2)*distance)+20,((j+1)*distance)+10),11)
#				pygame.draw.circle(window,(0,0,0),(((h+2)*distance)+20,((j+1)*distance)+10),10)

		#Drawing the output rectangles
		for h in xrange(0,self.outputs):
			if self.o_n[h] < 0: 			color = (255,0,0)
			elif self.o_n[h] == 0: 			color = (255,255,255)
			else: 							color = (0,255,0)
			pygame.draw.rect(window, color, (((self.hidden_layers+2)*distance)+20,(h+1)*distance,20,20),1)

		#Drawing the weights from input to hidden layer representing the weight with the thickness
		for h in xrange(0,self.inputs):
			for j in xrange(0,self.hidden_neurons-1):
				if self.w_ih[h][j] < 0: 	color = (255,0,0)
				elif self.w_ih[h][j] == 0: 	color = (255,255,255)
				else: 						color = (0,255,0)

				if self.w_ih[h][j] < 0:		thickness = int(round(self.w_ih[h][j]*-5))
				else:						thickness = int(round(self.w_ih[h][j]*5))
				pygame.draw.line(window, color, (distance+20, ((h+1)*distance)+10), (((0+2)*distance)+10, ((j+1)*distance)+10),thickness)

		#Drawing the weights from hidden to hidden
		if(self.hidden_layers > 1):
			for h in xrange(0,self.hidden_layers-1):
				print h 
				for j in xrange(0,self.hidden_neurons):
					for k in xrange(0,self.hidden_neurons-1):
						if self.w_hh[h][j][k] < 0: 		color = (255,0,0)
						elif self.w_hh[h][j][k] == 0: 	color = (255,255,255)
						else: 							color = (0,255,0)

						if self.w_hh[h][j][k] < 0:		thickness = int(round(self.w_hh[h][j][k]*-5))
						else:							thickness = int(round(self.w_hh[h][j][k]*5))
		#				pygame.draw.line(screen, (0, 0, 255), (0, 0), (200, 100))
						pygame.draw.line(window, color,(((2+h)*distance)+30,(((j+1)*distance)+10)),(((3+h)*distance)+10,((k+1)*distance)+10),thickness)

		#Drawing the weights from the last hidden layer the outputs
		for h in xrange(0,self.hidden_neurons):
			for j in xrange(0,self.outputs):
				if self.w_ho[h][j] < 0: 	color = (255,0,0)
				elif self.w_ho[h][j] == 0: 	color = (255,255,255)
				else: 						color = (0,255,0)

				if self.w_ho[h][j] < 0:		thickness = int(round(self.w_ho[h][j]*-5))
				else:						thickness = int(round(self.w_ho[h][j]*5))
#				pygame.draw.line(screen, (0, 0, 255), (0, 0), (200, 100))
				pygame.draw.line(window, color,(((self.hidden_layers+1)*distance)+30,(((h+1)*distance)+10)),(((self.hidden_layers+2)*distance)+20,((j+1)*distance)+10),thickness)

		pygame.display.update() 

		running = True;
		while(running):
			for event in pygame.event.get():
#				print event.type
				if event.type == 5:
					pygame.display.quit(); running = False; 