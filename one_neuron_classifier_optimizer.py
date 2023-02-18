#!/usr/bin/env python

##  one_neuron_classifier.py

"""
A one-neuron model is characterized by a single expression that you see in the value
supplied for the constructor parameter "expressions".  In the expression supplied, the
names that being with 'x' are the input variables and the names that begin with the
other letters of the alphabet are the learnable parameters.
"""

import random
import numpy

seed = 0           
random.seed(seed)
numpy.random.seed(seed)
#Setting a uniform learning rate
learn_rate = 1e-2

from ComputationalGraphPrimer import *

#Modify original class so loss_record could be plotted on same graph
class ComputationalGraphPrimer_mod(ComputationalGraphPrimer):
    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

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
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

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


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)     ## BACKPROP loss
        return loss_running_record

#Create a SGD+ subclass that would inherit from ComputationalGraphPrimer
class ComputationalGraphPrimer_SGDplus(ComputationalGraphPrimer_mod):
    #Specifying a momentum coefficient and initializing the first bias 
    def __init__(self, *args, **kwargs):
        ComputationalGraphPrimer.__init__(self, *args, **kwargs)
        self.momentum = 0.99
        self.bias_step = 0
        #Initializing a dictionary to store the gradients based on their learnable parameters
        self.learnable_gradients = {"ab":0, "bc":0, "cd":0, "ac":0}
    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params

        #Enumerate over each batch of learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            # Calculate the next step in the parameter hyperplane
            gradient = ((self.learning_rate * y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid) + self.momentum * self.learnable_gradients[param])
            ## Update the learnable parameters 
            self.vals_for_learnable_params[param] += gradient
            self.learnable_gradients[param] = gradient
        #Update bias based on momentum
        self.bias_step = self.bias_step *self.momentum + self.learning_rate * y_error * deriv_sigmoid    ## Update the bias
        self.bias += self.bias_step
    ######################################################################################################
    

#Adam subclass for Computational GraphPrimer that inherits from ComputationalGraphPrimer_mod
class ComputationalGraphPrimer_adam(ComputationalGraphPrimer_mod):
    #Specifying a momentum coefficient and initializing the first step size while inheriting from the superclass
    def __init__(self, *args, **kwargs):
        ComputationalGraphPrimer.__init__(self, *args, **kwargs)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8
        self.m = 0
        self.v = 0
        self.learnable_gradients = {"ab":0, "bc":0, "cd":0, "ac":0}

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params

        #Enumerate over each batch of learnable_params and optimize steps based on adam
        for i,param in enumerate(self.vals_for_learnable_params):
            # Calculate the first & second moment estimates on a running-average basis
            self.m = (self.beta1 * self.m) + ((1-self.beta1) * self.learnable_gradients[param])
            self.v = (self.beta2 * self.v) + ((1-self.beta2)*(self.learnable_gradients[param]**2))
            #Computing the bias-corrected estimate of m & v
            m_hat = self.m/(1-self.beta1**(i+1))
            v_hat = self.v/(1-self.beta2**(i+1))
            step = -self.learning_rate * (m_hat)/((v_hat + self.epsilon)**(1/2))
            
            ## Update the learnable parameters 
            self.vals_for_learnable_params[param] += step
            self.learnable_gradients[param] = step

        self.bias += self.learning_rate * y_error * deriv_sigmoid    ## Update the bias
    ######################################################################################################




#For original SGD without step-size optimizer
cgp = ComputationalGraphPrimer_mod(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
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
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = learn_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )
#For SGD_adam
cgp_adam = ComputationalGraphPrimer_adam(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = learn_rate,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

cgp.parse_expressions()
cgp_plus.parse_expressions()
cgp_adam.parse_expressions()

#cgp.display_network1()
#cgp.display_network2()
# cgp.display_one_neuron_network()      

training_data_original = cgp.gen_training_data()
training_data_plus = cgp_plus.gen_training_data()
training_data_adam = cgp_adam.gen_training_data()


#Calling on the methods to plot the graphs of loss_record
loss_running_record = cgp.run_training_loop_one_neuron_model( training_data_original)
loss_running_record_plus = cgp_plus.run_training_loop_one_neuron_model( training_data_plus )
loss_running_record_adam = cgp_adam.run_training_loop_one_neuron_model( training_data_adam )

#Plotting out the figures
plt.figure()
plt.plot(loss_running_record, label = 'SGP')
plt.plot(loss_running_record_plus, label = 'SGP+')
plt.plot(loss_running_record_adam, label = 'ADAM')
plt.legend()
plt.show()