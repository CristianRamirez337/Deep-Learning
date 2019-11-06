# Some Notes Taken Regarding Practice Questions on Neural Networks
## Detection algorithms
1. (Q3)You are working on a factory automation task. Your system will see a can of soft-drink coming down a conveyor belt, and you want it to take a picture and decide whether (i) there is a soft-drink can in the image, and if so (ii) its bounding box. Since the soft-drink can is round, the bounding box is always square, and the soft drink can always appears as the same size in the image. There is at most one soft drink can in each image.

   What is the most appropriate set of output units for your neural network?
   
   > Logistic unit, bx, by

2. (Q7)In the YOLO algorithm, at training time, only one cell ---the one containing the center/midpoint of an object--- is responsible for detecting this object.
   > True

## Face recognition and Neural style transfer
1. (Q1) Face verification requires comparing a new picture against one person’s face, whereas face recognition requires comparing a new picture against K person’s faces.
   > True
2. (Q2) Why do we learn a function d(img1, img2)d(img1,img2) for face verification? (Select all that apply.)
   > True: This allows us to learn to recognize a new person given just a single image of that person.
   
   > True: We need to solve a one-shot learning problem.
   
   > False: This allows us to learn to predict a person's identity using a softmax output unit, where the number of classes equals the number of persons in the database plus 1 (for the final "not in database" class).
3. (Q4) Which of the following is a correct definition of the triplet loss? Consider that α>0.
   > max(||f(A)−f(P)||2−||f(A)−f(N)||2+α,0)
4. (Q5) The upper and lower neural networks have different input images, but have exactly the same parameters.
   > True
5. (Q7) Neural style transfer is trained as a supervised learning task in which the goal is to input two images (xx), and train a network to output a new, synthesized image (yy).
   > False: images have no labels.

## Recurrent Neural Networks
1. (Q4) You are training this RNN language model. At the t-th time step, what is the RNN doing?
> Estimating P(y<t> | y<1>, y<2>, …, y<t-1>)
2. (Q5) You have finished training a language model RNN and are using it to sample random sentences. What are you doing at each time step tt?
> (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as y<t>. (ii) Then pass this selected word to the next time-step.
3. (Q7) Suppose you are training a LSTM. You have a 10000 word vocabulary, and are using an LSTM with 100-dimensional activations a<t>. What is the dimension of Γu at each time step?
> 100
3. (Q8) Alice proposes to simplify the GRU by always removing the Γu. I.e., setting Γu = 1. Betty proposes to simplify the GRU by removing the Γr. I. e., setting Γr = 1 always. Which of these models is more likely to work without vanishing gradient problems even when trained on very long input sequences?
> Betty’s model (removing Γr), because if Γu≈0 for a timestep, the gradient can propagate back through that timestep without much decay.

## Natural Language Processing & Word Embeddings
1. (Q2) What is t-SNE?
> A non-linear dimensionality reduction technique
2. (Q4) Which of these equations do you think should hold for a good word embedding? (Check all that apply)
> e_boy - e_girl ≈ e_brother - e_sister

> e_boy - e_brother ≈ e_girl - e_sister 
3. (Q5) Let E be an embedding matrix, and let o1234 be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don’t we call E * o1234 in Python?
> It is computationally wasteful.
4. (Q6) When learning word embeddings, we create an artificial task of estimating P(target∣context). It is okay if we do poorly on this artificial prediction task; the more important by-product of this task is that we learn a useful set of word embeddings.
> True
5. (Q7) In the word2vec algorithm, you estimate P(t∣c), where t is the target word and c is a context word. How are t and c chosen from the training set?
> c and t are chosen to be nearby words.
6. (Q8) Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The word2vec model uses the softmax function.
Which of these statements are correct? Check all that apply.
> (True) θt and ec are both 500 dimensional vectors.

> (True) θt and ec are both trained with an optimization algorithm such as Adam or gradient descent.

> (False) After training, we should expect θt to be very close to ec when t and c are the same word.
7. (Q9) Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings.The GloVe model minimizes this objective.
Which of these statements are correct? Check all that apply.
> (True) θi and ej should be initialized randomly at the beginning of training.

> (True) Xij is the number of times word i appears in the context of word j.

> (True) The weighting function f(.) must satisfy f(0) = 0.
