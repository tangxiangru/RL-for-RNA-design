class Model():
    """Base structure for all models"""
    def __init__(self):
        pass
   
    """You may populate model with (possibly hypothetical) data or nothing. Initial population
    of data may allow your model to generate fitness values faster, or anchor the model on experimental data"""
    def populate_model(self,data=None):
        pass

    """ This is your true model, given a sequence, it should generate its fitness. Can be a pretrained model on data.
     Should be treated as a private function."""
    def _fitness_function(self,sequence):    
        pass

    """This is a public wrapper on fitness function, it allows faster lookup, as well as, if you want, overriding your model with actual data"""
    def get_fitness(self,sequence):
        pass



    

    
