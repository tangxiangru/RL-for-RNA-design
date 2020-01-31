

#TODO fix the _ vs W reference for WT before adding machine learning/real data.

class Empirical_Regression(Model):

	"""This is an empirical model, where fitness values come from a regression model trained on experimental data.
	The @param use_empirical_values is used to indicate whether (when available) real values are used when returning 
	the fitness for a sequence, or the model's prediction"""
    def __init__(self,use_empirical_values=False):
        self.sequences={}
   
    """You must populate model with (possibly hypothetical) data or nothing. Initial population
    of data may allow your model to generate fitness values faster, or anchor the model on experimental data"""
    def populate_model(self,data):
        pass

    def learn_fitness_function_from_data(self,data):
    	pass

    """ This is your true model, given a sequence, it should generate its fitness. Can be a pretrained model on data.
     Should be treated as a private function."""
    def _fitness_function(self,sequence):    
        pass

    """This is a public wrapper on fitness function, it allows faster lookup, as well as, if you want, overriding your model with actual data"""
    def get_fitness(self,sequence):

    	if use_empirical_values and sequence in self.sequences:
    	   return self.sequences[sequence]
    	else:	
    	   self.sequences[sequence]=self._fitness_function(sequence)		
    	   return self.sequences[sequence]

