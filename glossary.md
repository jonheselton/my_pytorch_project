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
    <dt>Channel</dt>
    <dd>When dealing with image processing in ML, channels refer to the colors in the image.    an input with 3 channels has one channel each for Red, Green, and Blue.  A greyscale image has a single channel.</dd>
    <dt>Convolution</dt>
    <dd>The act of applying a convolutional filter to a matrix in order to train weights</dd>
    <dd>Referring to the convolutional layer, or a convolutional operation.</dd>
    <dt>Convolution Fileter</dt>
    <dd>A matrix with the same number of dimensions of the input data, but a smaller shape.  For example if the input image is 28x28 pixels, the filter could be any matrix smaller than 28x28</dd>
    <dt>Convolutional Layer</dt>
    <dd>A layer in a deep neural network that performs a series of convolutional operations on different slices of the input matrix</dd>
    <dt>Convolutional Operation</dt>
    <dd>An operation on the input matrix with a convolutional filter.  Element-wise multiplication of a slice of the input layer, the shape of which matches the filet, and then the individual elements of the resulting matrix are added together.</dd>
    <dt>RELU</dt>
    <dd>REctified Linear Unit</dd>
    <dd>A type of activation function</dd>
    <dd>If input is negative or 0, the output is 0.  If the input is positive, the output is equal to the input.</dd>
    <dt>Activation Function</dt>
    <dd></dd>
    <dt>Connections</dt>
    <dd></dd>
    <dt>Fully connected</dt>
    <dd></dd>
    <dt>Guasian layer</dt>
    <dd></dd>
    <dt>Loss Function</dt>
    <dd>Calculates loss, or error, based on the output of the NN and the target.</dd>
</dl>