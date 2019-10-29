# Some Random Notes Taken Regarding Neural Networks

## Very deep neural networks & ResNets
While a very deep neural network can represent very complex function and learn features at different levels of abstraction,
there exists a huge barrier to training a very deep NN: the vanishing gradients. 

During gradient descent, as you backprop from the final layer back to the first layer, you multiply by the weight matrix on
each step, and thus the gradient can decrease expoentially quickly to zero (or, in rare cases, grow exponentially quickly      and "explode" to take very large values), which makes the gradient decent prohibitively slow.

During training, you often see that the magnitude/norm (≈learning speed) of the gradient of the shallower layers decrease 
to zero rapidly as training proceeds.

## Identity block in Resnet
There is also some evidence that the ease of learning an identity function accounts for ResNets' remarkable performance 
even more so than skip connections helping with vanishing gradients.

## Convolutional block in ResNet
In the case when the input and output dimensions of a block don't match up, we can add a CONV layer (usually followed by
BatchNormalization) in the shortcut path to resize the input X to a different dimension.

The CONV layer on the shortcut path does not use any non-linear activation function, as its main goal is to apply a (learned)
linear function that resizes the dimension of the input.

## Object detection
### Classification with localization
#### Define target label y
y = [p_c, b_x, b_y, b_h, b_w, c1, c2, c3]

The loss function can be defined as either:
- squared error for all entries, or
- logistic regression loss for p_c, squared error for b_x, b_y, b_h, b_w, and log-likelihood loss for c1, c2, c3

#### Landmark detection
The landmark coordinates should be consistent across examples (e.g. l1 always represents 'noes tip').


### Convolutional implementation of sliding windows
A MaxPooling layer with a stride of s corresponds to a window slided by a stride of s.


### Anchor box
Choose anchor box shapes: can choose by hand or automatically by K-means.

Some unsolved problems:
- What if you have only two anchor boxes but detect three objects in a grid cell?
- Two objects in a grid cell that have the same anchor box shape?
