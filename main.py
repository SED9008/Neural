#!/usr/bin/python2

from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 	= [[0,0],[0,1],[1,0],[1,1]]
truth_out 	= [[0],[1],[1],[0]]

net 		= network(2,1,3,1) 				#inputs, hidden_layers, hidden_neurons, outputs
net.debug 	= True
net.alpha	= 1								#Learning rate

#net.trainEpochs(truth_in,truth_out,2000) 	#input_set, output_set, learning_rate, epochs
net.trainSSE(truth_in,truth_out,0.01)		#input_set, output_set, learning_rate, target_sse