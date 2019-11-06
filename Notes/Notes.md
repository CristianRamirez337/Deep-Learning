# Some Random Notes Taken Regarding Neural Networks

## ResNet
### Very deep neural networks & ResNets
While a very deep neural network can represent very complex function and learn features at different levels of abstraction,
there exists a huge barrier to training a very deep NN: the vanishing gradients. 
<br/>
During gradient descent, as you backprop from the final layer back to the first layer, you multiply by the weight matrix on
each step, and thus the gradient can decrease expoentially quickly to zero (or, in rare cases, grow exponentially quickly      and "explode" to take very large values), which makes the gradient decent prohibitively slow.

During training, you often see that the magnitude/norm (≈learning speed) of the gradient of the shallower layers decrease 
to zero rapidly as training proceeds.

### Identity block in Resnet
There is also some evidence that the ease of learning an identity function accounts for ResNets' remarkable performance 
even more so than skip connections helping with vanishing gradients.
<br/>

### Convolutional block in ResNet
In the case when the input and output dimensions of a block don't match up, we can add a CONV layer (usually followed by
BatchNormalization) in the shortcut path to resize the input X to a different dimension.
<br/>
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
<br/>

### Convolutional implementation of sliding windows
A MaxPooling layer with a stride of s corresponds to a window slided by a stride of s.
<br/>

### Anchor box
Choose anchor box shapes: can choose by hand or automatically by K-means.

Some unsolved problems:
- What if you have only two anchor boxes but detect three objects in a grid cell?
- Two objects in a grid cell that have the same anchor box shape?

### YOLO algorithm
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

## Face recognition
Some challenges of a face recognition system:
- Only one training example for each person
- Don't want to re-train the model every time new data (employee) is added into the database (softmax output doesn't work)

### Encoding
In Face Verification, you're given two images and you have to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person.

Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on.

By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

### Triplet loss
Why we want to carefully choose triplets for training?
- Randomly chosen triplets would be too "easy" for the network to learn, hence reduces computational efficiency.

## Neural style transfer
### Style matrix
In linear algebra, the Gram matrix G of a set of vectors (v1,…,vn) is the matrix of dot products, whose entries are  Gij=np.dot(vi,vj)

## Sequence data
Why not a traditional neural network?
- Inputs and outputs can be differents lengths in different examples.
> If we (zero) pad the input to be the length of the maximum length, then the total input size will be very large, resulting in a very large number of parameters.
- Doesn't share features learned across different positions of text.

### Pros and Cons for Character-level model
Pro:
- Get rid of the <UNK>

Con:
- Longer sequences
- Not as good as word-level models at capturing long-range dependencies between how early parts of a sentence affect later
  parts.
- Computationally expensive to train

## Recurrent Neural Network
Vanishing gradients: not able to capture long-range dependencies
> Gated recurrent unit (GRU)

Sometimes, exploding gradients (NaN) also happen
> Gradient clipping

## Word Embeddings and Transfer Learning
1. Learn word embeddings from large text corpus. (1-100B words)
   (Or download pre-trained embedding online.)
2. Transfer embedding to new task with smaller training set. (say, 100k words)
3. Optional: Continue to finetue the word embeddings with new data.

The last step is optional, and it's usually useful when the task has a pretty big data set. 
> When you try to transfer from some task A to some task B, the process of transfer learning is most useful when you happen to have a ton of data for A and a relatively small data set for B. And this is true for a lot of NLP tasks (e.g. name entity recognition, text summarization, co-reference resolution, parsing, etc.) and less true for some language modelling and machine translation.
### T-SNE mapping
T-SNE mapping is a complicated and highly non-linear process, and hence the analogical relationships that exist in the original high-dimensional space might or might not hold true after T-SNE mapping.

### Context & Target pairs
If your goal is to build a language model, it's natural to use, for example, the last four words as your context.
If your goal is to learn word embeddings, then many other simpler contexts can also do remarkably well.

### Skip-grams model
Problem: the softmax layer is computationally explensive, requiring to sum over all the vocab size in the denominator.
One possible solution: Hierarchical softmax, with each node as a binary logistic classification, reduces the computation cost from linear of |v| to log|v|. In practice, the hierarchical softmax tree is set up such that the more frequent words tend to be up in the tree while the less frequent words are burried deep downside the tree.

#### Negative sampling model
Instead of using a v-weight softmax classification, we now have v binary logistic regression classifications, and on each iteration, we choose 1 positive example & K negative examples, hence train (K+1) binary classifications.

How to select the negative examples?
> Choose a distribution that lies between the two extremes (i.e. empirical distribution and uniform distribution).

### GloVe model
Goal: learn vectors whose inner product is a good predictor for how often two words occur together.

### A note on featurization view of word embeddings
For learned word vectors, it's difficult to look at individual components and assign human interpretation, as the linear transformation can be arbitrary, not well-aligned with humanly interpretable axes, and sometimes even not orthogonal (?).

However, despite the arbitrary linear transformation, the parallelogram map for describing analogy still works.

### Sentiment classification
By using an embedding matrix that is learned from a much larger text corpus, you can take knowledge from even not frequent words, and apply/generalize to you classification problem, even for words not in the labeled training set.

## Beam search
An approximate/heuristic search algorithm.
### Length normalization
The product of probabilities can be too small that results in underfloating, i.e. being too small for the floating part representation to be stored accurately by computer.
> By taking log, we end up with a more numerically stable algorithm, less prone to rounding error.

The long product of probabilities make the algorithm prefer shorter sentences.

## Error analysis
Go through dev set and choose examples where the algorithm produces a much worse output than human, and then ascribe problem to either the search algorithm (Beam search) or the objective function (RNN).

When comparing P(y*|x) with P(y^|x), remember to take into consideration length normalization.
> Take average of the probability of each word by Ty, i.e. length of the sentence (softer approach: Ty^alpha).

If RNN is to blame,
> Regularization, adding training data, different network architecture, ... (See Course 3)

## Bleu score
Human generated references would be provided as part of the dev set or as part of the test set.

## Attention model
One disadvantage: the computation cost is quadratic (Tx * Ty).

Applications of Attention models: machine translation, image captioning, etc.
