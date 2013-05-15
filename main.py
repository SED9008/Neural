#!/usr/bin/python2

from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 		= [[0],[1],[2],[3]]
truth_out 		= [[0,0,0],[0,0,1],[0,1,0],[0,1,1]]

net 			= network(1,2,4,3) 					#inputs, hidden_layers, hidden_neurons, outputs
net.debug 		= True
net.alpha		= 2									#Learning rate

net.useGraph()

net.graphFreq	= 50
#net.graph 		= False

#net.train(truth_in,truth_out,0,2000) 			#input_set, output_set, learning_rate, mode, epochs
cnt = net.train(truth_in,truth_out,1,0.01)		#input_set, output_set, learning_rate, mode, target_sse
#net.showNet(True,cnt)