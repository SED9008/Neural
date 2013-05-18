#!/usr/bin/python2

from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 			= [[0,0],[0,1],[1,0],[1,1]]
truth_out 			= [[0],[1],[1],[0]]

net 				= network(2,2,4,1) 					#inputs, hidden_layers, hidden_neurons, outputs
net.debug 			= True
net.alpha			= 5									#Learning rate
net.adaptive_alpha	= True
net.alpha_roof		= 10

net.useGraph()

net.graphFreq		= 50
#net.graph 			= False

#net.train(truth_in,truth_out,0,2000) 			#input_set, output_set, learning_rate, mode, epochs
cnt = net.train(truth_in,truth_out,1,0.01)		#input_set, output_set, learning_rate, mode, target_sse
#net.showNet(True,cnt)