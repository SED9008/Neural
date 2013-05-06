#!/usr/bin/python2

import time
import random
from network import network
import pprint

pp = pprint.PrettyPrinter(indent=4)


truth_in 	= [[0,0],[0,1],[1,0],[1,1]]
truth_out 	= [[0],[1],[1],[0]]

net = network(2,1,2,1) #inputs,hidden_layers,hidden_neurons,outputs
print "--------------------------"
net.initWeights()
pp.pprint(net.weights)
print "------------------------"
net.weights = [[[0.5,0.9],[0.4,1.0],[0.8,-0.1]],[[-1.2],[1.1],[0.3]]]
pp.pprint(net.weights)
print "------------------------"
errors = 10
cnt = 0
random.seed()
choice_list = []
for h in xrange(len(truth_in)):
	choice_list.append(h)
samples = 1
#for h in xrange(0,len(truth_in)):
while(errors > 0.1):
	errors = 0
	for j in xrange(0,len(truth_in)):
		net.calcOutputs(truth_in[j])
		net.calcErrorGrad(truth_out[j])
		net.adjustWeights(0.2)
		errors += net.sse
		
#		pp.pprint(net.neural_outputs)
#		print "------------------------"
#		pp.pprint(net.weights)
#		print "------------------------"
		
		print truth_in[j], truth_out[j], net.neural_outputs[net.hidden_layers+1][net.outputs-1], net.error, net.sse
#		raw_input()
#		print "########################"
	cnt += 1
print "count", cnt
print "samples", samples
samples += 1

errors = 10


for h in xrange(len(truth_in)):
	net.calcOutputs(truth_in[h])
	print truth_in[h], truth_out[h], net.neural_outputs[net.hidden_layers+1][net.outputs-1]
