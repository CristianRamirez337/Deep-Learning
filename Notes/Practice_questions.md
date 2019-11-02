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
