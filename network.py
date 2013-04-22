#!/usr/bin/python2
import random
import math
import pprint
import pygame	
pp = pprint.PrettyPrinter(indent=4)

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
		self.w_ih	= [[random.uniform(-0.5,0.5) for h in xrange(self.hidden_neurons-1)]for h in xrange(self.inputs)]
		#Weights from hidden layer to hidden layers
		if(self.hidden_layers > 1):
			self.w_hh	= [[[random.uniform(-0.5,0.5) for h in xrange(self.hidden_neurons-1)] for h in xrange(self.hidden_neurons)] for h in xrange(hidden_layers-1)]
		#Weights from last hidden neurons to the output neurons
		self.w_ho 	= [[random.uniform(-0.5,0.5) for h in xrange(self.outputs)] for h in xrange(self.hidden_neurons)]		
#		print self.w_ih,"\n--------"
#		print len(self.w_ih)
#		print self.w_hh
#		pp.pprint(self.w_ho)
#		print len(self.w_ho)

#		print self.h_n

	"""Calculate the output of every neuron using the sigmoid function
	Could be extended with a choide in step functions"""
	def activate(self):
#		pp.pprint(self.h_n)
		#From input to the first hidden layer
		for h in xrange(0,self.hidden_neurons-1):
			#zeroing the neuron output
			self.h_n[0][h]	= 0	
			for j in xrange(0,self.inputs):
				self.h_n[0][h] += self.i_n[j]*self.w_ih[j][h]
#				print h,j,self.h_n[0][h],"|",self.i_n[j],"*",self.w_ih[j][h]
			self.h_n[0][h] = self.sigmoid(self.h_n[0][h])
#		print self.h_n
#		pp.pprint(self.h_n)
#		print "--------------------"

		#From hidden to hidden of there are more than 1 hidden layer
		if(self.hidden_layers > 1):
#			pp.pprint(self.h_n)
			#For each hidden layer
			for h in xrange(0,self.hidden_layers-1):
				#For each hidden neuron input
#				print h,"="
				for j in xrange(0,self.hidden_neurons-1):
					#zeroing the neuron output
#					print j,"-"
					self.h_n[h+1][j] = 0;
					#For each hidden neuron that has to be calculated
					#The bias of the next layer is always -1 and does not have to be calculated
					for k in xrange(0,self.hidden_neurons-1): 
#						print k
#						print self.w_hh[h][k][j]
						self.h_n[h+1][j] += self.h_n[h][j]*self.w_hh[h][k][j]
					self.h_n[h+1][j] = self.sigmoid(self.h_n[h+1][j])
#			print self.h_n
#			pp.pprint(self.h_n)
#		print "--------------------"
#		pp.pprint(self.o_n)
		#From last hidden layer to the output layer
		for h in xrange(0,self.outputs):
			#zeroing the neuron output
			self.o_n[h]	= 0	
			for j in xrange(0,self.hidden_neurons):
				self.o_n[h] += self.h_n[self.hidden_layers-1][j]*self.w_ho[j][h]
#				print h,j,self.h_n[0][h],"|",self.i_n[j],"*",self.w_ih[j][h]
			self.h_n[0][h] = self.sigmoid(self.h_n[0][h])
#		pp.pprint(self.o_n)

	def sigmoid(self,x):
		return 1/(1+math.exp(x))

	def step(self, x):	
		if (x >= 0):
			return 1
		elif(x < 0):
			return 0	
	def properties(self):
		print "Inputs:",self.inputs,"\nHidden layers:",self.hidden_layers,"\nHidden neurons:",self.hidden_neurons,"\nOutputs:",self.outputs

	def showNet(self):

		pygame.init() 
		distance = 80
		window = pygame.display.set_mode((((self.hidden_layers+2)*distance)+distance+(distance/2),((self.hidden_neurons+1)*distance)+20))
#		window = pygame.display.set_mode(((self.hidden_layers+2)*(2*distance)), ((self.hidden_neurons+1)*distance+20)) 

		""""Drawing the iputs, hidden neurons and ouput neurons"""
		for h in xrange(0,self.inputs):
			pygame.draw.rect(window, (255,255,255), (distance,(h+1)*distance,20,20),1)

		for h in xrange(0,self.hidden_layers):
			for j in xrange(0,self.hidden_neurons):
				pygame.draw.circle(window,(255,255,255),(((h+2)*distance)+20,((j+1)*distance)+10),11)
		#		pygame.draw.circle(window,(0,0,0),(((h+2)*distance)+20,((j+1)*distance)+10),10)

		for h in xrange(0,self.outputs):
			pygame.draw.rect(window, (255,255,255), (((self.hidden_layers+2)*distance)+20,(h+1)*distance,20,20),1)

		for h in xrange(0,self.inputs):
			for j in xrange(0,self.hidden_neurons):
				pygame.draw.line(window, (255, 255, 255), (distance+20, ((h+1)*distance)+10), (((0+2)*distance)+20, ((j+1)*distance)+10))
#				pygame.draw.lines(window, (255,255,255), True, ((distance,(h+1)*distance),(((j+1)*distance)+10)), 1)
		
		pygame.display.update() 

		running = True;
		while(running):
			for event in pygame.event.get():
#				print event.type
				if event.type == 5:
					pygame.display.quit(); running = False; 


net = network(2,1,2,1)
net.properties()
#net.activate()
net.showNet()
