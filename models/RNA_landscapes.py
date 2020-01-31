import sys
sys.path.append('/usr/local/ViennaRNA/lib/python3.7/site-packages/')
# sys.path.append("/n/home01/ssinaei/sw/lib/python3.4/site-packages/")

import RNA
from meta.model import Model
import numpy as np

class RNA_landscape(Model):
    def __init__(self,wt,threshold=False,noise=0):
        self.wt=wt
        self.sequences={}
        self.noise=noise
        self.threshold=threshold

    def _fitness_function(self,sequence):
        _,fe=RNA.fold(sequence)
   
        if self.threshold!=False:
           if -fe>self.threshold: 
              return 1
           else:
              return 0

        return -fe/85

    def get_fitness(self,sequence):
         if self.noise==0:
           if sequence in self.sequences:
              return self.sequences[sequence]
           else:
              self.sequences[sequence]=self._fitness_function(sequence)
              return self.sequences[sequence]
         else:
              self.sequences[sequence]=self._fitness_function(sequence)+np.random.normal(scale=self.noise)
         return self.sequences[sequence]



class RNA_landscape_Binding(Model):
    def __init__(self,target, threshold=False,noise=0,norm_value=1):
        self.target=target
        self.sequences={}
        self.threshold=threshold
        self.noise=noise
        self.norm_value=norm_value

    
    def _fitness_function(self,sequence):
        duplex=RNA.duplexfold(self.target,sequence)
        fitness=-duplex.energy
        if self.threshold!=False:
           if fitness>self.threshold: 
              return 1
           else:
              return 0

        return fitness/self.norm_value

    def get_fitness(self,sequence):
         if self.noise==0:
           if sequence in self.sequences:
              return self.sequences[sequence]
           else:
              self.sequences[sequence]=self._fitness_function(sequence)
              return self.sequences[sequence]
         else:
              self.sequences[sequence]=self._fitness_function(sequence)+np.random.normal(scale=self.noise)

    
