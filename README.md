Neural Network in Python
========================
(Under construction)

This object will allow you to create and train artificial neural networks. This project spawned to help myself learn the language python.
A nice feature is that the network can be graphically represented using the pygame module. This gives a lot of insight in what is actually happening to the network.

My goal is to get familiarised with the different techniques that can be used to create and train a neural network. 
Over time functions and tricks will be added to see how they affect speed and stability of learning.



network.py
==========

network(self, inputs, hidden_layers, hidden_neurons, outputs)
-------------------------------------------------------------
This function creates the network with the desired parameters.
It also initialises the basic variable structures and initialises the weights.

initWeights(self)
-----------------
Initialises the weights using the random module from python.

calcOuts(self,inputs)
---------------------
Calculates the outputs for every single neuron according to a certain input set that has to be passed along.

calcErrors(self,desired)
------------------------
Calculates the errors and error gradients so we can backpropegate the error through the network. This is done by using the desired output that has to be passed along.

adjustWeights(self)
-------------------
This function adjusts the weights according to the previously calculated error gradients

trainEpochs(self, input_set, output_set, epochs)
------------------------------------------------
Trains the weights for a certain amount of epochs(cycles) using a specified in- and output set.

trainSSE(self, input_set, output_set, target_sse)
-------------------------------------------------
Trains the weights until the sum of the squared errors is lower than the specified target SSE. Again using a specified in- and output set.

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

showNet(self, hold, epoch)
--------------------------
This function draws the neural network graphically to give a neat insight in what is happening to the network.
I have not done any graphical programming so i'm just going to keep this as simple as possible. 

Warning: If your network is rather large this function will spawn an equally large window, it has no max size yet so be carefull!

getColor(self,x)
----------------
Function to decide what color a weight line or neuron box should get.
I decided green if positive, white if close to zero and red for negative.

getThickness(self,weight)
-------------------------
This function makes sure the weight is converted to int so pygame can use it to draw a line and it makes sure it cant get thicker than a certain maximum.