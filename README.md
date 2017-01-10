# ml-contest-aranz
Investigation into machine learning as part of the geological contest

2016-ml contest 

https://github.com/seg/2016-ml-contest/blob/master/index.ipynb 
I will be using google's tensorflow library to try a variety of machine learning solutions to the problem. This choice is due to the flexibility and resources available with this library. It is also described as Numpy on steroids which should give some familiarity to myself. 
The goal is, given a csv file of wells with various measurements at each depth, predict the correct facie value for each depth. 
 
INSTALLATION 
Install on windows using the anaconda package manager. This seems to be the only way to get all the dependencies to work properly. The installation guide provided by google is reasonable. 
https://www.tensorflow.org/get_started/os_setup#anaconda_installation 
 
BENCHMARK 
There are 10 inputs and 9 potential facie values in our model. By random guess we should expect to get ~11% of answers correct (this is our lowest benchmark that all models must beat). The benchmark set by the contest is 41%. The current highest result seems to be around 60% accuracy. 
 
DATA CLEANUP 
In order to use the csv files data we need to ensure that the input columns are regularised. In this case I have replaced the formation and well names with integer values, this will allow it to be an input into our model (as we cannot input a string). We also normalize the fields and replace missing values. 
The output (facie value) is also changed into a one-hot encoding rather than an integer, this allows for the result to be a probability of the correct facie. The result is then computed by taking the max probability index. 
 
IMPLEMENTATION 
As a first pass we use the train/test data sets given from the competition. We also copy the very basic MINST implementation given in the tensorflow tutorial. This consists of a basic regression matrix of the form:  
y = x.W + b  
Where W is the weights and b is the biases. These values are updated each iteration by the regression algorithm. We update by minimizing the cross-entropy between our predicted results y and our actual results y_. 
Each iteration is calculated on the entire training set and a halting condition is given when the accuracy difference between 500 iterations has not changed by more than 0.001. 
The accuracy of our results are defined as the performance of the algorithm in classifying the test set correctly. 
First pass 
As a first pass the basic MINST tutorial implementation was used. This uses a very basic gradient decent algorithm to calculate the weights.   
Using the GradientDecentOptimizer to minimize the cross-entropy and a learning rate of 1e-3 produced: ~19% accuracy with non-convergence. 
Learning rate 
Non-convergence indicates that we cannot find a local minimum, this can be due to a variety of reasons but a common one is that our learning rate is too high. Metaphorically this is that we are constantly jumping around with large steps meaning that we overshoot the minimum. 
A basic solution to this is to reduce the learning rate, effectively taking smaller steps with each gradients change. 
Reducing the learning rate to 1e-6 resulted in convergence and a final accuracy of 22%.  
Optimisation algorithm 
22% is very poor. One way to improve this is to use a different optimisation algorithm. Gradient decent is a rather limited algorithm and often struggles to find the global minimum. Taking cues from other contestants we instead will try the ADAM optimizer https://arxiv.org/pdf/1412.6980.pdf. This is a more refined algorithm that implements exponential decay rate for the estimates. Using a Adam optimiser with a learning rate of 1e-5 over 20,000 iterations produced an accuracy of 45%.  
The accuracy after 20,000 iterations seemed to be continuing to improve, thus I increased the number of iterations to 120,000. This produced an accuracy of 48%.  
Training and test data randomization 
The initial training/test data is not randomised. Rather it is split by well name (and ordered by depth). This will lead to overfitting of our algorithm. To combat this, we add the two datasets together and then randomise the index (using a seed for reproducibility). Then we split the data into 20% training, 80% test. This should ensure that overfitting is not too significant. After running over 15,000 iterations with a learning rate of 1e-3 the accuracy was ~49%, the training data accuracy was larger at ~63%. 
Although this isn't much of an improvement in accuracy, the resulting gradients solution will be much less overfit compared to the training/test sets given. This gives us improved confidence in the validity of our accuracy results. Compare this to the LA team using neural networks, wherein their training results where far higher than the official result on a blind dataset (~80% compared to 56% actual). 
Use random initial weights 
Currently we are using zeros as our initial weight biases, instead we can initialise them to be random. This noise should help in finding the global minimum. The result of this was an unchanged accuracy, however the first epoch often had a very low accuracy compared to an initialisation of zeros (which is expected if the solution weights are mainly zero). 
Deep network 
Let's go deep. http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html  
The topography of the hidden layers in neural networks are a tricky problem and there is no real consensus for deciding. One point that seems to be agreed upon is that the more hidden layers in the network, the harder it becomes to train (for a variety of reasons). So as a first pass we will try with one hidden layer of n= 9 (based on the rule of thumb that the hidden layer size should be bounded by the input and output feature numbers). 
 
n == 10, @ 1e-3 and 20,000 iterations: 51%, training set accuracy 84%.  
n == 15 @1e-3 and 20,000 iterations: 50%, training set accuracy 90%.  
The training accuracy is now looking like a reasonable figure (and would probably be a good submission if we could get the test accuracy to the same level). It also improves as we increase the number of parameters in the hidden layer. This is to be expected, as we are increasing the complexity of the model and thus we expect it to be able to model the training set almost perfectly. However as a result we are now experience significant overfitting in our model. This means that the system will not be able to generalise well to new data; the whole point. 
On way to reduce overfitting is to implement dropout on the network and/or increase the ratio of test/training data. 
Training ratios 
Increasing the ratio to 60/40:test/train. This means that we will not be overfitting so much to our current dataset, but we do run the risk of producing an undergeneralized solution.  
n == 10, @ 1e-3 and 20,000 iterations: 55%, training set accuracy 70%.  
As expected this has brought the test and training sets closer. Looking at some prior examples, it looks as though my ratio of test/train (and validation) should have an even larger ratio: http://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio. Using the approximate rule of thumb of:  
1/sqrt(# inputs) 
There should be 30% test data and 70% training, not 60%, or 80%. Using a ratio of 70/30:train/test, and as we have more data in our training set let's try a 2-layer neural network. 
n1 == 25, n2 == 15 @1e-3 and 20,000 iterations, 78%, training set accuracy 84%.  
Now we're getting somewhere! As expected the test accuracy has shot up to be much closer to the training accuracy. 
Batch normalisation 
 
 
