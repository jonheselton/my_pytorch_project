# My ML Glossary

*These defenitions may be rewritten in my own words and should not be assumed to be technically or factually accurate*

<dl>
    <dt>Tensor</dt>
    <dd>Similiar to arrays and matrices.  The PyTorch class includes convenient methods
    </dd>
    <dt>Neural Network (nn)</dt>
    <dd>Made up of nested functions.  The functions are defined by weighs and biases</dd>
    <dt>Weight</dt>
    <dd><dd>A value that is output (learned) during the training process, and used as an input when making predictions
    </dd>
    <dt>Bias (mathematical)</dt>
    <dd>A value indicating the offset from the origin.  In a 2D curve, it is the value of Y when X = 0</dd>
    <dt>Forward Propegation</dt>
    <dd>Running an input through the NN to arrive at a prediction</dd>
    <dt>Backward Propegation</dt>
    <dd>Takes the loss and updates the gradients of each layer, which in turn adjust the weights in the NN based on the learning rate.</dd>
    <dt>Learning rate</dt>
    <dd>A float that controls how much parameters are changed by during gradient descent.  Too low and the model takes too long, too high and convergence may not be reached.</dd>
    <dt>Convergence</dt>
    <dd>The point at which changes in error become so small as to be negligible</dd>
    <dt>Momentum</dt>
    <dd>A gradient descent algo that takes into account the derivitives of preceeding steps in addition to the current step.</dd>
    <dt>Gradient descent</dt>
    <dd>A mathematical technique to minimize loss. Gradient descent iteratively adjusts weights and biases, gradually finding the best combination to minimize loss.</dd>
    <dt>Affine operation</dt>
    <dd></dd>
    <dt>Channel</dt>
    <dt>Convolution</dt>
    <dt>RELU activation function</dt>
    <dt>Connections</dt>
    <dt>Fully connected</dt>
    <dt>Guasian layer</dt>
    <dt>Loss Function</dt>
    <dd>Calculates loss, or error, based on the output of the NN and the target.</dd>
</dl>