#!/usr/bin/python2
import random
import math
import pprint
import pygame

pp = pprint.PrettyPrinter(indent=4)

class network(object):
	def __init__(self,inputs,hidden_layers,hidden_neurons,outputs):
		self.rounding = 6
		self.inputs			= inputs + 1
		self.hidden_layers	= hidden_layers
		self.hidden_neurons	= hidden_neurons + 1
		self.outputs		= outputs

		self.layer_neurons	= []
		self.layer_weights	= []
		for h in xrange(0,self.hidden_layers + 2):
			self.layer_neurons.append([])
			self.layer_weights.append([])

		self.layer_neurons[0] 					= self.inputs
		self.layer_neurons[1] 					= self.hidden_neurons
		self.layer_neurons[self.hidden_layers+1]= self.outputs
		self.layer_weights[0] 					= self.inputs
		self.layer_weights[1] 					= self.hidden_neurons
		self.layer_weights[self.hidden_layers+1]= self.outputs + 1
		
		if(self.hidden_layers > 1):
			for h in xrange(2,self.hidden_layers+1):
				self.layer_neurons[h] 			= self.hidden_neurons
				self.layer_weights[h] 			= self.hidden_neurons 

		#creating the ouput structure
		self.outs = []
		for h in xrange(0,len(self.layer_neurons)):
			self.outs.append([])
			for j in xrange(0,self.layer_neurons[h]):
				self.outs[h].append([])

		#initialising bias
		self.outs[0][self.inputs-1] = -1
		if(self.hidden_layers > 1):
			for h in xrange(1,self.hidden_layers+1):
				self.outs[h][self.hidden_neurons-1] 	= -1
		else:
			self.outs[1][self.hidden_neurons-1] 		= -1

		self.delta = []
		for h in xrange(0,len(self.layer_neurons)-1):
			self.delta.append([])
			for j in xrange(0,self.layer_weights[h+1]-1):
				self.delta[h].append([])

	def initWeights(self):
		random.seed()
		self.weights = []
		#Create the layers in the weights structure
		for h in xrange(0,len(self.layer_weights)-1):
			self.weights.append([])
			for j in xrange(0,self.layer_weights[h]):
				self.weights[h].append([])
				for k in xrange(0,self.layer_weights[h+1]-1):
					self.weights[h][j].append(round(random.uniform(-1,1),2))

	def calcOutputs(self,inputs):
		for h in xrange(self.inputs-1):
			self.outs[0][h] = inputs[h] 

		for h in xrange(1,len(self.layer_weights)):
			for j in xrange(0,self.layer_weights[h]-1):
				self.outs[h][j] = 0;
				for k in xrange(0,self.layer_weights[h-1]):
					self.outs[h][j] += self.outs[h-1][k] * self.weights[h-1][k][j]
				self.outs[h][j] = round(self.activate(self.outs[h][j]),self.rounding)

	def calcErrors(self,desired):
		sse = 0
		for h in xrange(0,self.outputs):
			error = desired[h] - self.outs[self.hidden_layers+1][h]
			self.delta[self.hidden_layers][h] = self.outs[self.hidden_layers+1][h] * (1-self.outs[self.hidden_layers+1][h]) * error
			sse += error * error

		for h in xrange(0,self.hidden_neurons-1):
			self.delta[self.hidden_layers-1][h] = 0
			for j in xrange(0,self.outputs):
				self.delta[self.hidden_layers-1][h] += self.delta[self.hidden_layers][j] * self.weights[self.hidden_layers][h][j]
			self.delta[self.hidden_layers-1][h] = self.outs[self.hidden_layers][h] * (1-self.outs[self.hidden_layers][h]) * self.delta[self.hidden_layers-1][h]
		
		if(self.hidden_layers > 1):
			for h in xrange(self.hidden_layers-2,-1,-1):
				for j in xrange(0,self.hidden_neurons-1):
					self.delta[h][j] = 0
					for k in xrange(0,self.hidden_neurons-1):
						self.delta[h][j] += self.delta[h+1][k] * self.weights[h][j][k]
					self.delta[h][j] = self.outs[h+1][h] * (1-self.outs[h+1][h]) * self.delta[h][j]
		return sse

	def adjustWeights(self, alpha):
		for h in xrange(len(self.delta)-1,-1,-1):
			for j in xrange(0,self.layer_neurons[h]):
				for k in xrange(0,self.layer_weights[h+1]-1):
					self.weights[h][j][k] += alpha * self.outs[h][j] * self.delta[h][k]

	#Using this to maybe add some more activation functions 
	#and study the way they work
	def activate(self,x):
		return self.sigmoid(x)
	def sigmoid(self,x):
		return (1/(1+math.exp(-x)))
	def hypTangent(self,x):
		#a & b Chosen by Guyon, 1991
		a = 1.716
		b = 0.667
		return ((2*a)/(1+math.exp(-b*x)))-a

	def showNet(self, hold):
		pygame.init() 
		dist = 100
		rect 	 = 20
		height_neurons = self.inputs+1
		if(self.inputs > self.hidden_neurons):
			height_neurons = self.inputs+1
		else:
			height_neurons = self.hidden_neurons
		window = pygame.display.set_mode(((len(self.layer_neurons))*dist,(height_neurons)*dist))

		for h in xrange(0,len(self.layer_neurons)-1):
			for j in xrange(0,self.layer_weights[h]):
				for k in xrange(0,self.layer_weights[h+1]-1):
					thickness 	= self.getThickness(self.weights[h][j][k])
					color 		= self.getColor(self.weights[h][j][k])
					point1 = [(h*dist)+(dist/2)+(rect/2),(j*dist)+(dist/2)]
					point2 = [((h+1)*dist)+(dist/2)-(rect/2),(k*dist)+(dist/2)]
#					print point1,point2
					pygame.draw.line(window, color, point1, point2, thickness)

		for h in xrange(0,len(self.layer_neurons)):
			for j in xrange(0,self.layer_neurons[h]):
				color = self.getColor(self.outs[h][j])
				pygame.draw.rect(window, color, (((h*dist)+(dist/2)-(rect/2)),((j*dist)+(dist/2)-(rect/2)),rect,rect),2)
		
		pygame.display.update()
		
		 

		"""
		for h in xrange(0,self.inputs):
			if self.i_n[h] < 0: 			color = (255,0,0)
			elif self.i_n[h] == 0: 			color = (255,255,255)
			else: 							color = (0,255,0)
			pygame.draw.rect(window, color, (distance,(h+1)*distance,20,20),1)

		#Drawing the hidden layer circles
		for h in xrange(0,self.hidden_layers):
			for j in xrange(0,self.hidden_neurons):
				if self.h_n[h][j] < -0.1: 	color = (255,0,0)
				elif self.h_n[h][j] > 0.1: 	color = (0,255,0)
				else: 						color = (255,255,255)
				pygame.draw.circle(window,color,(((h+2)*distance)+20,((j+1)*distance)+10),11)
		#				pygame.draw.circle(window,(0,0,0),(((h+2)*distance)+20,((j+1)*distance)+10),10)

		#Drawing the output rectangles
		for h in xrange(0,self.outputs):
			if self.o_n[h] < -0.1: 			color = (255,0,0)
			elif self.o_n[h] > 0.1: 		color = (0,255,0)
			else: 							color = (255,255,255)
			pygame.draw.rect(window, color, (((self.hidden_layers+2)*distance)+20,(h+1)*distance,20,20),1)

		#Drawing the weights from input to hidden layer representing the weight with the thickness
		for h in xrange(0,self.inputs):
			for j in xrange(0,self.hidden_neurons-1):
				weight 		= self.weights[0][h][j]
				color 		= self.getColor(weight)
				thickness 	= self.getThickness(int(round(weight)))
				pygame.draw.line(window, color, (distance+20, ((h+1)*distance)+10), (((0+2)*distance)+10, ((j+1)*distance)+10),thickness)
		
		#Drawing the weights from hidden to hidden
		if(self.hidden_layers > 1):
			for h in xrange(1,self.hidden_layers): #changed some shit here
		#				print h 
				for j in xrange(0,self.hidden_neurons):
					for k in xrange(0,self.hidden_neurons-1):
						weight = int(round(self.weights[h][j][k]))
						color 		= self.getColor(weight)
						thickness 	= self.getThickness(weight)
		#				pygame.draw.line(screen, (0, 0, 255), (0, 0), (200, 100))
						pygame.draw.line(window, color,(((1+h)*distance)+30,(((j+1)*distance)+10)),(((2+h)*distance)+10,((k+1)*distance)+10),thickness)
		
		#Drawing the weights from the last hidden layer the outputs
		for h in xrange(0,self.hidden_neurons):
			for j in xrange(0,self.outputs):
				weight = int(round(self.weights[self.hidden_layers][h][j]))
				color 		= self.getColor(weight)
				thickness 	= self.getThickness(weight)
		#				pygame.draw.line(screen, (0, 0, 255), (0, 0), (200, 100))
				pygame.draw.line(window, color,(((self.hidden_layers+1)*distance)+30,(((h+1)*distance)+10)),(((self.hidden_layers+2)*distance)+20,((j+1)*distance)+10),thickness)

		pygame.display.update() 
		"""
		if hold == True:
			running = True;
			while(running):
				for event in pygame.event.get():
		#				print event.type
					if event.type == 5:
						pygame.display.quit(); running = False; 

	def getColor(self,x):
		if(x < 0): 		return (255,0,0)
		elif(x == 0): 	return (255,255,255)
		else: 				return (0,255,0)

	def getThickness(self,weight):
		thickness_multi = 1
		if weight < 0:	return int(weight*-thickness_multi)
		else:			return int(weight*thickness_multi)