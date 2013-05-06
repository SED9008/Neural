#!/usr/bin/python2
import random
import math
import pprint

pp = pprint.PrettyPrinter(indent=4)

class network(object):
	def __init__(self,inputs,hidden_layers,hidden_neurons,outputs):
		self.inputs			= inputs + 1
		self.hidden_layers	= hidden_layers
		self.hidden_neurons	= hidden_neurons + 1
		self.outputs		= outputs
		self.layer_neurons	= []
		self.layer_weights	= []
		for h in xrange(0,self.hidden_layers + 2):
			self.layer_neurons.append([])
			self.layer_weights	.append([])

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

#		pp.pprint(self.layer_neurons)
#		pp.pprint(self.layer_weights)
		#creating the ouput structure
		self.neural_outputs = []
		for h in xrange(0,len(self.layer_neurons)):
			self.neural_outputs.append([])
			for j in xrange(0,self.layer_neurons[h]):
				self.neural_outputs[h].append([])
#		pp.pprint(self.neural_outputs)

		#initialising bias
		self.neural_outputs[0][self.inputs-1] = -1
		if(self.hidden_layers > 1):
			for h in xrange(1,self.hidden_layers+1):
				self.neural_outputs[h][self.hidden_neurons-1] 	= -1
		else:
			self.neural_outputs[1][self.hidden_neurons-1] 		= -1
#		pp.pprint(self.neural_outputs)

	def initWeights(self):
		random.seed()
		self.weights = []
#		pp.pprint(self.weights)	
		#Create the layers in the weights structure
		for h in xrange(0,len(self.layer_weights)-1):
			self.weights.append([])
			for j in xrange(0,self.layer_weights[h]):
				self.weights[h].append([])
				for k in xrange(0,self.layer_weights[h+1]-1):
					self.weights[h][j].append(round(random.uniform(-1,1),2))
#		pp.pprint(self.weights)	

	def calcOutputs(self,inputs):
		if len(inputs) != self.inputs-1:
			print "False inputs"
			return False

		for h in xrange(self.inputs-1):
			self.neural_outputs[0][h] = inputs[h] 
#		pp.pprint(self.neural_outputs)
		for h in xrange(1,len(self.layer_weights)):
			for j in xrange(0,self.layer_weights[h]-1):
				self.neural_outputs[h][j] = 0;
				for k in xrange(0,self.layer_weights[h-1]):
#					print h,j,k
					self.neural_outputs[h][j] += self.neural_outputs[h-1][k] * self.weights[h-1][k][j]
				self.neural_outputs[h][j] = self.activate(self.neural_outputs[h][j])
#		pp.pprint(self.neural_outputs)

	def calcErrorGrad(self,desired_out):
		if len(desired_out) != self.outputs:
			print "False output data"
			return False

		#Error gradient delta k
		self.delta_k = [0 for h in xrange(self.outputs)]
		#Error gradient delta j
		self.delta_j = [[0 for h in xrange(self.hidden_neurons-1)] for h in xrange(self.hidden_layers)]
		#Sum of squared error
		self.sse = 0
		multi = 1
		self.error = 0
		for h in xrange(0,self.outputs):
			self.error 		= desired_out[h] - self.neural_outputs[self.hidden_layers+1][h]
			self.delta_k[h] = self.neural_outputs[self.hidden_layers][h] * (1 - self.neural_outputs[self.hidden_layers][h]) * self.error * multi
			self.sse 		= self.error * self.error
#		print "error -", error
#		pp.pprint(self.delta_k)

		for h in xrange(0,self.hidden_neurons-1):
			self.delta_j[self.hidden_layers-1][h] = 0;
			for j in xrange(0,self.outputs):
				self.delta_j[self.hidden_layers-1][h] += self.delta_k[j] * self.weights[self.hidden_layers][h][j]
			self.delta_j[self.hidden_layers-1][h] = self.neural_outputs[self.hidden_layers][h] * (1 - self.neural_outputs[self.hidden_layers][h]) * self.delta_j[self.hidden_layers-1][h]

		if(self.hidden_layers > 1):
			for h in xrange(self.hidden_layers-2,-1,-1):
				for j in xrange(0,self.hidden_neurons-1):
					self.delta_j[h][j] = 0;
					for k in xrange(0,self.hidden_neurons-1):
						self.delta_j[h][j] += self.delta_j[h+1][k] * self.weights[h+1][j][k]	
					self.delta_j[h][j] = self.neural_outputs[h+1][j] * (1 - self.neural_outputs[h+1][j]) * self.delta_j[h][j]
#		pp.pprint(self.delta_j)

	def adjustWeights(self,learning_rate):
#		pp.pprint(self.weights)
		for h in xrange(0,self.hidden_layers):
			for j in xrange(0,self.layer_weights[h]):
				for k in xrange(0,self.layer_weights[h+1]-1):
#					print h,j,k
					self.weights[h][j][k] += learning_rate * self.neural_outputs[h][j] * self.delta_j[h][k]

		for h in xrange(0,self.hidden_neurons):
			for j in xrange(0,self.outputs):
#				print "-",self.hidden_layers,h,j
				self.weights[self.hidden_layers][h][j] += learning_rate * self.neural_outputs[self.hidden_layers][h] * self.delta_k[j]
#		pp.pprint(self.weights)

	def activate(self,x):
		return self.sigmoid(x)

	#Hyperbolic tangent activation function
	def hypTangent(self,x):
		#a & b Chosen by Guyon, 1991
		a = 1.716
		b = 0.667
		return ((2*a)/(1+math.exp(-b*x)))-a

	def sigmoid(self,x):
		return (1/(1+math.exp(-x)))




		def showNet(self, hold):
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
		if hold == True:
			running = True;
			while(running):
				for event in pygame.event.get():
	#				print event.type
					if event.type == 5:
						pygame.display.quit(); running = False; 

	def getColor(self,weight):
		if weight < 0: 		return (255,0,0)
		elif weight == 0: 	return (255,255,255)
		else: 				return (0,255,0)

	def getThickness(self,weight):
		thickness_multi = 1
		if weight < 0:	return int(weight*-thickness_multi)
		else:			return int(weight*thickness_multi)