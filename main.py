#!/usr/bin/python2

import time
import random
from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 	= [[0,0],[0,1],[1,0],[1,1]]
truth_out 	= [[0],[1],[1],[0]]

net = network(2,1,2,1)
net.initWeights()
sse = 10;
#h = 1
cnt = 0
while(sse > 0.01):
	sse = 0
	for h in xrange(0,len(truth_in)):
		net.calcOutputs(truth_in[h])
		sse += net.calcErrors(truth_out[h])
		net.adjustWeights(0.1)
		print truth_in[h],truth_out[h],net.outs[net.hidden_layers+1],sse
		cnt += 1
print cnt
pp.pprint(net.weights)

for h in xrange(len(truth_in)):
	net.calcOutputs(truth_in[h])
	print truth_in[h],truth_out[h],net.outs[net.hidden_layers+1]
