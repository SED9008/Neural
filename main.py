#!/usr/bin/python2

import time
import random
from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 	= [[0,0],[0,1],[1,0],[1,1]]
truth_out 	= [[0],[1],[1],[0]]

learning_rate = 1

net = network(2,2,2,1) #inputs, hidden_layers, hidden_neurons, outputs
net.initWeights()
pp.pprint(net.weights)
sse = 10;
#h = 1
cnt = 0




while(sse > 0.01):
	sse = 0
	for h in xrange(0,len(truth_in)):
		net.calcOutputs(truth_in[h])
#		pp.pprint(net.delta)
		sse += net.calcErrors(truth_out[h])
#		pp.pprint(net.delta)
#		raw_input()
		net.adjustWeights(learning_rate)
		print truth_in[h],truth_out[h],net.outs[net.hidden_layers+1],sse
		net.showNet(False)
		cnt += 1
#		pp.pprint(net.weights)
#		raw_input()
		
print "Epochs:",cnt/4, "Learning_rate of:", learning_rate,"\n"
pp.pprint(net.weights)

print ""
for h in xrange(len(truth_in)):
	net.calcOutputs(truth_in[h])
	print truth_in[h],truth_out[h],net.outs[net.hidden_layers+1]
net.showNet(True)