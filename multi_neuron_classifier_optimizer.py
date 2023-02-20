#!/usr/bin/env python

##  multi_neuron_classifier.py

"""
The main point of this script is to demonstrate saving the partial derivatives during the
forward propagation of data through a neural network and using that information for
backpropagating the loss and for updating the values for the learnable parameters.  The
script uses the following 4-2-1 network layout, with 4 nodes in the input layer, 2 in
the hidden layer and 1 in the output layer as shown below:


                               input

                                 x                                             x = node

                                 x         x|                                  | = sigmoid activation
                                                     x|
                                 x         x|

                                 x

                             layer_0    layer_1    layer_2


To explain what information is stored during the forward pass and how that
information is used during the backprop step, see the comment blocks associated with
the functions

         forward_prop_multi_neuron_model()   
and
         backprop_and_update_params_multi_neuron_model()

Both of these functions are called by the training function:

         run_training_loop_multi_neuron_model()

"""

import random
import numpy

seed = 0           
random.seed(seed)
numpy.random.seed(seed)
#Setting a uniform learning rate
learn_rate = 1e-2

from ComputationalGraphPrimer import *
#Modify original run_training_loop_multi_neuron_model method to plot all the loss_records on the same graph
class ComputationalGraphPrimer_mod(ComputationalGraphPrimer):
  def run_training_loop_multi_neuron_model(self, training_data):
    
    class DataLoader:
        """
        To understand the logic of the dataloader, it would help if you first understand how 
        the training dataset is created.  Search for the following function in this file:

                          gen_training_data(self)
        
        As you will see in the implementation code for this method, the training dataset
        consists of a Python dict with two keys, 0 and 1, the former points to a list of 
        all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
        the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
        classes have different means and variances.  The dimensionality of each data sample
        is set by the number of nodes in the input layer of the neural network.

        The data loader's job is to construct a batch of samples drawn randomly from the two
        lists mentioned above.  And it mush also associate the class label with each sample
        separately.
        """
        def __init__(self, training_data, batch_size):
            self.training_data = training_data
            self.batch_size = batch_size
            self.class_0_samples = [(item, 0) for item in self.training_data[0]]    ## Associate label 0 with each sample
            self.class_1_samples = [(item, 1) for item in self.training_data[1]]    ## Associate label 1 with each sample

        def __len__(self):
            return len(self.training_data[0]) + len(self.training_data[1])

        def _getitem(self):    
            cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                        ##   samples to be chosen randomly from the two lists
            if cointoss == 0:
                return random.choice(self.class_0_samples)
            else:
                return random.choice(self.class_1_samples)            

        def getbatch(self):
            batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
            maxval = 0.0                                               ## For approximate batch data normalization
            for _ in range(self.batch_size):
                item = self._getitem()
                if np.max(item[0]) > maxval: 
                    maxval = np.max(item[0])
                batch_data.append(item[0])
                batch_labels.append(item[1])
            batch_data = [item/maxval for item in batch_data]          ## Normalize batch data       
            batch = [batch_data, batch_labels]
            return batch                


    """
    The training loop must first initialize the learnable parameters.  Remember, these are the 
    symbolic names in your input expressions for the neural layer that do not begin with the 
    letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
    over the interval (0,1).
    """
    self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

    self.bias = [random.uniform(0,1) for _ in range(self.num_layers-1)]      ## Adding the bias to each layer improves 
                                                                              ##   class discrimination. We initialize it 
                                                                              ##   to a random number.

    data_loader = DataLoader(training_data, batch_size=self.batch_size)
    loss_running_record = []
    i = 0
    avg_loss_over_iterations = 0.0                                          ##  Average the loss over iterations for printing out 
                                                                              ##    every N iterations during the training loop.   
    for i in range(self.training_iterations):
        data = data_loader.getbatch()
        data_tuples = data[0]
        class_labels = data[1]
        self.forward_prop_multi_neuron_model(data_tuples)                                  ## FORW PROP works by side-effect 
        predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]      ## Predictions from FORW PROP
        y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]  ## Get numeric vals for predictions
        loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ## Calculate loss for batch
        loss_avg = loss / float(len(class_labels))                                         ## Average the loss over batch
        avg_loss_over_iterations += loss_avg                                              ## Add to Average loss over iterations
        if i%(self.display_loss_how_often) == 0: 
            avg_loss_over_iterations /= self.display_loss_how_often
            loss_running_record.append(avg_loss_over_iterations)
            print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))            ## Display avg loss
            avg_loss_over_iterations = 0.0                                                ## Re-initialize avg-over-iterations loss
        y_errors = list(map(operator.sub, class_labels, y_preds))
        y_error_avg = sum(y_errors) / float(len(class_labels))
        self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels)      ## BACKPROP loss
    return loss_running_record  

#Create SGD+ subclass through modifying back propagation method of multi neuron model
class ComputationalGraphPrimer_SGDplus(ComputationalGraphPrimer_mod):
  #Specifying a momentum coefficient and initializing the first bias 
  def __init__(self, *args, **kwargs):
    ComputationalGraphPrimer.__init__(self, *args, **kwargs)
    self.momentum = 0.99
    self.bias_step = 0
    #Initializing a dictionary to store the gradients based on their learnable parameters
    self.learnable_gradients = {"cp":0, "cq":0, "ap":0, "aq":0, "ar":0, "as":0, "bp":0, "bq":0, "br":0, "bs":0}

  def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels):
    """
    First note that loop index variable 'back_layer_index' starts with the index of
    the last layer.  For the 3-layer example shown for 'forward', back_layer_index
    starts with a value of 2, its next value is 1, and that's it.

    Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
    the samples in a batch.  To explain how this averaging is carried out by the
    backprop function, consider the last node on the example shown in the forward()
    function above.  Standing at the node, we look at the 'input' values stored in the
    variable "input_vals".  Assuming a batch size of 8, this will be list of
    lists. Each of the inner lists will have two values for the two nodes in the
    hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
    these values 'input vals' and store those in the variable "input_vals_avg".  Next we
    must carry out the same batch-based averaging for the partial derivatives stored in the
    variable "deriv_sigmoid".

    Pay attention to the variable 'vars_in_layer'.  These store the node variables in
    the current layer during backpropagation.  Since back_layer_index starts with a
    value of 2, the variable 'vars_in_layer' will have just the single node for the
    example shown for forward(). With respect to what is stored in vars_in_layer', the
    variables stored in 'input_vars_to_layer' correspond to the input layer with
    respect to the current layer. 
    """
    # backproped prediction error:
    pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
    pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
    for back_layer_index in reversed(range(1,self.num_layers)):
      input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
      input_vals_avg = [sum(x) for x in zip(*input_vals)]
      input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
      deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
      deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
      deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                        [float(len(class_labels))] * len(class_labels)))
      vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
      vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

      layer_params = self.layer_params[back_layer_index]         
      ## note that layer_params are stored in a dict like        
          ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
      ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
      transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

      backproped_error = [None] * len(vars_in_next_layer_back)
      for k,varr in enumerate(vars_in_next_layer_back):
          for j,var2 in enumerate(vars_in_layer):
            backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                          pred_err_backproped_at_layers[back_layer_index][i] 
                                          for i in range(len(vars_in_layer))])
#                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
      pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
      input_vars_to_layer = self.layer_vars[back_layer_index-1]
      for j,var in enumerate(vars_in_layer):
        layer_params = self.layer_params[back_layer_index][j]
        ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3 
        ##  lecture slides for how the parameters are updated using the partial derivatives stored away 
        ##  during forward propagation of data. The theory underlying these calculations is presented 
        ##  in Slides 68 through 71. 
        for i,param in enumerate(layer_params):
          gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] 
          step = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j] + (self.momentum * self.learnable_gradients[param])
          self.vals_for_learnable_params[param] += step
          self.learnable_gradients[param] = step
      #Update bias based on momentum
      self.bias_step = self.bias_step * self.momentum + self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) \
                                                                      * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)
      self.bias[back_layer_index-1] += self.bias_step
######################################################################################################

#Create Adam subclass through modifying ComputationalGraph Primer back propagation method
class ComputationalGraphPrimer_adam(ComputationalGraphPrimer_mod):
  #Specifying a momentum coefficient and initializing the first bias 
  def __init__(self, *args, **kwargs):
    ComputationalGraphPrimer.__init__(self, *args, **kwargs)
    self.beta1 = 0.9
    self.beta2 = 0.99
    self.epsilon = 1e-3
    self.m = 0
    self.v = 0

    #Initializing a dictionary to store the gradients based on their learnable parameters
    self.learnable_gradients = {"cp":0, "cq":0, "ap":0, "aq":0, "ar":0, "as":0, "bp":0, "bq":0, "br":0, "bs":0}


  def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels):
    """
    First note that loop index variable 'back_layer_index' starts with the index of
    the last layer.  For the 3-layer example shown for 'forward', back_layer_index
    starts with a value of 2, its next value is 1, and that's it.

    Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
    the samples in a batch.  To explain how this averaging is carried out by the
    backprop function, consider the last node on the example shown in the forward()
    function above.  Standing at the node, we look at the 'input' values stored in the
    variable "input_vals".  Assuming a batch size of 8, this will be list of
    lists. Each of the inner lists will have two values for the two nodes in the
    hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
    these values 'input vals' and store those in the variable "input_vals_avg".  Next we
    must carry out the same batch-based averaging for the partial derivatives stored in the
    variable "deriv_sigmoid".

    Pay attention to the variable 'vars_in_layer'.  These store the node variables in
    the current layer during backpropagation.  Since back_layer_index starts with a
    value of 2, the variable 'vars_in_layer' will have just the single node for the
    example shown for forward(). With respect to what is stored in vars_in_layer', the
    variables stored in 'input_vars_to_layer' correspond to the input layer with
    respect to the current layer. 
    """
    # backproped prediction error:
    pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
    pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
    for back_layer_index in reversed(range(1,self.num_layers)):
      input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
      input_vals_avg = [sum(x) for x in zip(*input_vals)]
      input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
      deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
      deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
      deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                        [float(len(class_labels))] * len(class_labels)))
      vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
      vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

      layer_params = self.layer_params[back_layer_index]         
      ## note that layer_params are stored in a dict like        
          ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
      ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
      transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

      backproped_error = [None] * len(vars_in_next_layer_back)
      for k,varr in enumerate(vars_in_next_layer_back):
          for j,var2 in enumerate(vars_in_layer):
            backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                          pred_err_backproped_at_layers[back_layer_index][i] 
                                          for i in range(len(vars_in_layer))])
  #                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
      pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
      input_vars_to_layer = self.layer_vars[back_layer_index-1]
      for j,var in enumerate(vars_in_layer):
        layer_params = self.layer_params[back_layer_index][j]
        ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3 
        ##  lecture slides for how the parameters are updated using the partial derivatives stored away 
        ##  during forward propagation of data. The theory underlying these calculations is presented 
        ##  in Slides 68 through 71. 
        for i,param in enumerate(layer_params):
          # Calculate the first & second moment estimates on a running-average basis
          self.m = (self.beta1 * self.m) + ((1-self.beta1) * self.learnable_gradients[param])
          self.v = (self.beta2 * self.v) + ((1-self.beta2)*(self.learnable_gradients[param]**2))
          # gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j]
          # gradient = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j] 

          #Computing the bias-corrected estimate of m & v
          m_hat = self.m/(1-self.beta1**(i+1))
          v_hat = self.v/(1-self.beta2**(i+1))
          step = -self.learning_rate * (m_hat)/((v_hat)**(1/2) + self.epsilon)
          #Update the learnable parameters
          self.vals_for_learnable_params[param] += step
          self.learnable_gradients[param] = step

      self.bias[back_layer_index-1] += self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) \
                                                                           * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)
    ######################################################################################################

#Original SGD
cgp = ComputationalGraphPrimer_mod(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learn_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )
#For SGD+
cgp_plus = ComputationalGraphPrimer_SGDplus(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learn_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

#Adam
cgp_adam = ComputationalGraphPrimer_adam(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = learn_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

cgp.parse_multi_layer_expressions()
cgp_plus.parse_multi_layer_expressions()
cgp_adam.parse_multi_layer_expressions()

#cgp.display_network1()
#cgp.display_network2()
# cgp.display_multi_neuron_network()   

training_data = cgp.gen_training_data()
training_data_plus = cgp_plus.gen_training_data()
training_data_adam = cgp_adam.gen_training_data()

loss_running_record = cgp.run_training_loop_multi_neuron_model( training_data )
loss_running_record_plus = cgp_plus.run_training_loop_multi_neuron_model ( training_data_plus )
loss_running_record_adam = cgp_adam.run_training_loop_multi_neuron_model ( training_data_adam )

#Plotting out the graph
plt.figure()
plt.plot(loss_running_record, label = 'SGD')
plt.plot(loss_running_record_plus, label = 'SGD+')
plt.plot(loss_running_record_adam, label = 'Adam')
plt.legend()
plt.show()