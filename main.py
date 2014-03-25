#!/usr/bin/python2

from network import network

import pprint

truth_in 			= [[0,0],[0,1],[1,0],[1,1]]
truth_out 			= [[0],[1],[1],[0]]

net	                = network(2,1,3,1) 					#inputs, hidden_layers, hidden_neurons, outputs

net.initWeights()
#net.loadWeights("comp_gen_dice.txt")

net.debug 			= True
net.alpha			= 1									#Learning rate
net.adaptive_alpha	= True
net.alpha_roof		= 2

#net.calcOuts(truth_in[2])
#print net.outs[2]

net.useGraph()
net.graphFreq       = 1

#net.train(truth_in,truth_out,0,2000) 					#input_set, output_set, mode, epochs
#print "Training"
cnt = net.train(truth_in,truth_out,1,0.01)				#input_set, output_set, mode, target_sse
#net.showNet(True,)

#net.saveWeights("real_dice.txt")
#print "Saved weights"

#net.showNet(True,0)
#net.calcOuts(truth_in[0])
#print net.outs[2]