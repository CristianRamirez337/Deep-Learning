Some Random Notes Taken Regarding Neural Networks

1. While a very deep neural network can represent very complex function and learn featurs at different levels of abstraction,
there exists a huge barrier to training a very deep NN: the vanishing gradients. 

   During gradient descent, as you backprop from the final layer back to the first layer, you multiply by the weight matrix on
   each step, and thus the gradient can decrease expoentially quickly to zero (or, in rare cases, grow exponentially quickly      and "explode" to take very large values), which makes the gradient decent prohibitively slow.

   During training, you often see that the magnitude/norm (≈learning speed) of the gradient of the shallower layers decrease 
   to zero rapidly as training proceeds.
