#!/usr/bin/python2
import random
import math
import pprint
import pygame
import time

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

		self.debug	= False
		self.alpha	= 1	
		
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

		self.initWeights()

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

	def calcOuts(self, inputs):
		for h in xrange(self.inputs-1):
			self.outs[0][h] = inputs[h] 

		for h in xrange(1,len(self.layer_weights)):
			for j in xrange(0,self.layer_weights[h]-1):
				self.outs[h][j] = 0;
				for k in xrange(0,self.layer_weights[h-1]):
					self.outs[h][j] += self.outs[h-1][k] * self.weights[h-1][k][j]
				self.outs[h][j] = round(self.activate(self.outs[h][j]),self.rounding)

	def calcErrors(self, desired):
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
				for j in xrange(0,self.layer_weights[h+1]-1):
					self.delta[h][j] = 0
					for k in xrange(0,self.layer_weights[h+1]-1):
						self.delta[h][j] += self.delta[h+1][k] * self.weights[h+1][j][k]
					self.delta[h][j] = self.outs[h+1][j] * (1-self.outs[h+1][j]) * self.delta[h][j]
		return sse

	def adjustWeights(self):
		for h in xrange(len(self.delta)-1,-1,-1):
			for j in xrange(0,self.layer_neurons[h]):
				for k in xrange(0,self.layer_weights[h+1]-1):
					self.weights[h][j][k] += self.alpha * self.outs[h][j] * self.delta[h][k]

	def trainEpochs(self, input_set, output_set, epochs):
		if(len(input_set) != len(output_set)):
			print "Input and output set length mismatch!"
			return 0
		cnt = 0

		while(cnt < epochs):
			self.sse = 0
			for h in xrange(0,len(input_set)):
				self.calcOuts(input_set[h])
				self.sse += self.calcErrors(output_set[h])
				self.adjustWeights()
				self.showNet(False,cnt)
				if self.debug:
					print input_set[h],output_set[h],self.outs[self.hidden_layers+1],self.sse, cnt
			cnt +=1
		print ""
		pp.pprint(self.weights)
		print "\nEpochs:",cnt, "Learning_rate of:", self.alpha,"\n\n"

		for h in xrange(len(input_set)):
			self.calcOuts(input_set[h])
			print input_set[h],output_set[h],self.outs[self.hidden_layers+1]
			self.showNet(False,cnt)
			time.sleep(1)

		self.showNet(True,cnt)

	def trainSSE(self, input_set, output_set, target_sse):
		if(len(input_set) != len(output_set)):
			print "Input and output set length mismatch!"
			return 0
		cnt 		= 0
		self.sse 	= 10 #begin value to enter the whileloop
		while(self.sse > target_sse):
			self.sse = 0
			for h in xrange(0,len(input_set)):
				self.calcOuts(input_set[h])
				self.sse += self.calcErrors(output_set[h])
				self.adjustWeights()
				self.showNet(False,cnt)
				if self.debug:
					print input_set[h],output_set[h],self.outs[self.hidden_layers+1],self.sse, cnt
			cnt +=1
		print ""
		pp.pprint(self.weights)
		print "\nEpochs:",cnt, "Learning_rate of:", self.alpha,"\n\n"

		for h in xrange(len(input_set)):
			self.calcOuts(input_set[h])
			print input_set[h],output_set[h],self.outs[self.hidden_layers+1]
			self.showNet(False,cnt)
			time.sleep(1)

		self.showNet(True,cnt)


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

	def showNet(self, hold, epoch):
		pygame.init() 
		dist = 100
		rect 	 = 20
		height_neurons = self.inputs+1
		if(self.inputs > self.hidden_neurons):
			height_neurons = self.inputs+1
		else:
			height_neurons = self.hidden_neurons
		window = pygame.display.set_mode(((len(self.layer_neurons))*dist,(height_neurons)*dist))
		pygame.display.set_caption("Neural Network")
		window.fill((94,130,167))
		myfont = pygame.font.SysFont("DejaVuSans Mono", 15)
#		myfont.set_bold(True)
		text_height = 5

		label = myfont.render("Inputs", 8, (0,0,0))
		window.blit(label, (rect+2, text_height))
		label = myfont.render("Hidden", 8, (0,0,0))
		window.blit(label, (((self.hidden_layers+2)*dist)/2-28, text_height))
		label = myfont.render("Outputs", 8, (0,0,0))
		window.blit(label, (((self.hidden_layers+2)*dist)-(dist/2)-(1.5*rect), text_height))
		epoch = "Epoch: " + str(epoch)
		label = myfont.render(epoch, 8, (0,0,0))
		window.blit(label, (rect+2, dist*(height_neurons-1)+((dist/4)*3)))
		sse = "SSE: " + str(round(self.sse,3))
		label = myfont.render(sse, 8, (0,0,0))
		
		window.blit(label, (((self.hidden_layers+2)*dist)-(dist/2)-(3.25*rect), dist*(height_neurons-1)+((dist/4)*3)))

		for h in xrange(0,len(self.layer_neurons)-1):
			for j in xrange(0,self.layer_weights[h]): 
				for k in xrange(0,self.layer_weights[h+1]-1):
					thickness 	= self.getThickness(self.weights[h][j][k])
					color 		= self.getColor(self.weights[h][j][k])
					point1 = [(h*dist)+(dist/2)+(rect/2),(j*dist)+(dist/2)]
					point2 = [((h+1)*dist)+(dist/2)-(rect/2),(k*dist)+(dist/2)]
					pygame.draw.line(window, color, point1, point2, thickness)
#					pygame.draw.aaline(window, color, point1, point2, True)

		myfont = pygame.font.SysFont("monospace", 13)
		for h in xrange(0,len(self.layer_neurons)):
			for j in xrange(0,self.layer_neurons[h]):
				color = self.getColor(self.outs[h][j])
				label = myfont.render(str(round(self.outs[h][j])), 8, (0,0,0))
				window.blit(label, ((h*dist)+(dist/2)-(rect/2)-2,(j*dist)+(dist/2)+(rect/2)+(rect/8)))
				pygame.draw.rect(window, (0,0,0), (((h*dist)+(dist/2)-(rect/2)),((j*dist)+(dist/2)-(rect/2)),rect,rect))
				pygame.draw.rect(window, color, (((h*dist)+(dist/2)-(rect/2)),((j*dist)+(dist/2)-(rect/2)),rect,rect),3)

		pygame.display.update()

		if hold == True:
			running = True;
			while(running):
				for event in pygame.event.get():
#					print event.type
					if event.type == 12:
						pygame.display.quit(); running = False; 

	def getColor(self,x):
		if(x < 0): 					return (255,0,0)
		elif(x < 0.2 and x >= 0): 	return (255,255,255)
		else: 						return (0,255,0)

	def getThickness(self,weight):
		thickness_multi = 1.1
		max_width		= 10
		if weight < 0:				thickness = int(weight*-thickness_multi)
		else:						thickness = int(weight*thickness_multi)
		if thickness > max_width:	return max_width
		else:						return thickness