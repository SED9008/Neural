#!/usr/bin/python2

import time
from network import network
import pprint
pp = pprint.PrettyPrinter(indent=4)


truth_in 	= [[0,0],[0,1],[1,0],[1,1]]
truth_out 	= [[0],[1],[1],[0]]

#print len(truth_in)
"""
net = network(2,1,2,1)
hold = False
net.sse = 10
#while net.sse > 0.001:
x = 0
while net.sse > 0.01:
	for h in xrange(0,len(truth_in)):
		net.activate(truth_in[h])
		net.backPropegate(truth_out[h],0.2)
		net.showNet(False)
#		time.sleep(1)
#		print net.o_n
#		print truth_out[h]
#		print "--------------"
	x += 1
	if x > 50:
		print net.sse
		x = 0



for h in xrange(0,len(truth_in)):
	net.activate(truth_in[h])
	print net.i_n
	print net.o_n
print net.sse
net.showNet(True)


#net.properties()

#net.activate()
#net.showNet()
"""
net = network(2,2,2,1)
net.initWeights()
print "---------------"
pp.pprint(net.weights)
print "---------------"
net.activate(truth_in[0])
net.backPropegate(truth_out[0],0.1)
print "---------------"
pp.pprint(net.weights)
print "---------------"
net.showNet(True)