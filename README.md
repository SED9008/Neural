Neural Network in Python2
=========================
(Under construction)

This object will allow you to create and train artificial neural networks. This project spawned to help myself learn the language python.
A nice feature is that the network can be graphically represented using the pygame module. This gives a lot of insight in what is actually happening to the network. You do have to install the pygame module to be able to use this

My goal is to get familiarised with the different techniques that can be used to create and train a neural network. 
Over time functions and tricks will be added to see how they affect speed and stability of learning.

main.py
=======
This file shows how to create and train a network.
It is actually the file i use to study certain techniques and their effect and also to debug the network code. It should always show how to create and train a network though.
I will add an example in the readme to show how to use the object.

network.py
==========

network(self, inputs, hidden_layers, hidden_neurons, outputs)
-------------------------------------------------------------
This function creates a network object with the desired parameters.
It also initialises the basic variable structures and initialises the weights.

ex: net = network(2,2,3,1)

initWeights(self)
-----------------
Initialises the weights using the random module from python. This function is executed when creating the network.

ex: net.initWeights

calcOuts(self,inputs)
---------------------
Calculates the outputs for every single neuron according to a certain input set that has to be passed along.

ex: net.calcOuts(inputs)

calcErrors(self,desired)
------------------------
Calculates the errors and error gradients so we can backpropegate the error through the network. This is done by using the desired output that has to be passed along.

ex: net.calcErrors(desired)

adjustWeights(self)
-------------------
This function adjusts the weights according to the previously calculated error gradients

ex: net.adjustWeights()

trainEpochs(self, input_set, output_set, epochs)
------------------------------------------------
Trains the weights for a certain amount of epochs(cycles) using a specified in- and output set.

net.trainEpochs(input_set, output_set, 2000)

trainSSE(self, input_set, output_set, target_sse)
-------------------------------------------------
Trains the weights until the sum of the squared errors is lower than the specified target SSE. Again using a specified in- and output set.

net.trainSSE(input_set, output_set, 0.01)

showNet(self, hold, epoch)
--------------------------
This function draws the neural network graphically to give a neat insight in what is happening to the network.
I have not done any graphical programming so i'm just going to keep this as simple as possible. 
Closing the windows is possible by pressing alt+f4.
Hold is a boolean and when true, the graphical representation will be shown untill it is manually closed.
The variable epoch will show the number passed inside the graphical representation.

Warning: If your network is rather large this function will spawn an equally large window, it has no max size yet so be carefull!

ex: net.showNet(True, epoch)

Help Functions
==============
These functions are more under the hood functions. Check them out in the source code if you want to know more.

activate(self,x)
----------------
In this function it is possible to specify what activation functions for the neuron outputs has to be used. I specify this here so that i won't need to change is in every line where some neuron is activated.
My intentions are to make different activation function to study their effect.

sigmoid(self,x)
---------------
Applies the sigmoidal function to x as activation function

hypTangent(self,x)
------------------
Applies the hyperbolic tangent function to x as activation function

getColor(self,x)
----------------
Function to decide what color a weight line or neuron box should get.
I decided green if positive, white if close to zero and red for negative.

getThickness(self,weight)
-------------------------
This function makes sure the weight is converted to int so pygame can use it to draw a line and it makes sure it cant get thicker than a certain maximum.