#!/usr/bin/python2

from network import network

import pprint

truth_in 			= [[0,0],[0,1],[1,0],[1,1]]			#Input set
truth_out 			= [[0],[1],[1],[0]]					#Desired output set

net	                = network(2,1,3,1) 					#inputs, hidden_layers, hidden_neurons, outputs

net.initWeights()
#net.loadWeights("real_dice.txt")						#Load the weight values from a text file

net.debug 			= True								#Print debug info in the terminal
net.alpha			= 1									#Learning rate
net.adaptive_alpha	= True								#Adjust learning rate to increase if learning stagnates
net.alpha_roof		= 2 								#Learning rate max value

net.useGraph()											#Use the pygame visualisation
net.graph_freq      = 1									#Display the network after every epoch
#net.graph_image_seq	= True

#net.train(truth_in,truth_out,0,2000) 					#input_set, output_set, mode, epochs
cnt = net.train(truth_in,truth_out,1,0.01)				#input_set, output_set, mode, target_sse

#net.saveWeights("real_dice.txt")						#Save the weight values to a text file