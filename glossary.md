# My ML Glossary

*These defenitions may be rewritten in my own words and should not be assumed to be technically or factually accurate*
term
: definition

Tensors 
: Similiar to arrays and matrices.  The PyTorch class includes convenient methods
Neural Networks (NN) 
: Made up of nested functions.  The functions are defined by weighs and biases
Weight
: A value that is output (learned) during the training process, and used as an input when making predictions
Bias (mathematical)
: A value indicating the offset from the origin.  In a 2D curve, it is the value of Y when X = 0
Forward Propegation
: Running an input through the NN to arrive at a prediction
Backward Propegation
: Takes the loss and updates the gradients of each layer, which in turn adjust the weights in the NN based on the learning rate.
Learning rate
: A float that controls how much parameters are changed by during gradient descent.  Too low and the model takes too long, too high and convergence may not be reached.
Convergence
: The point at which changes in error become so small as to be negligible
Momentum
: A gradient descent algo that takes into account the derivitives of preceeding steps in addition to the current step.
Gradient descent
: A mathematical technique to minimize loss. Gradient descent iteratively adjusts weights and biases, gradually finding the best combination to minimize loss.
Affine operation
: 
Channel
:
Convolution
:
RELU activation function
:
Connections
:
Fully connected
:
Guasian layer
:
Loss Function
: Calculates loss, or error, based on the output of the NN and the target.