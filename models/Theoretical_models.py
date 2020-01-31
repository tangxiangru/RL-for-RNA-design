import numpy as np
from meta.model import Model
from utils.sequence_utils import *
import editdistance


class House_of_cards(Model):
    """House of cards model has random fitnesses everywhere"""
    def __init__(self):
        self.sequences={} #cache of current sequences with computed fitness

    def populate_model(self,data=None):
        for sequence in data:
            self.sequences[sequence]=np.random.standard_normal()
    
    def _fitness_function(self,sequence):
        return np.random.standard_normal()

    """Returns a fitness for a sequence """
    def get_fitness(self,sequence):
         if sequence in self.sequences:
            return self.sequences[sequence]
         else:
            self.sequences[sequence]=self._fitness_function(sequence)
            return self.sequences[sequence]

    def __str__(self):
        return "HoC"


class Simple_Additive(Model):
    """Additive fitness from singles + randomness determined by the ruggedness parameter"""

    def __init__(self,wt_full_seq, ruggedness=0.0,kill_switch=0.0,eta=1):
        self.sequences={} #cache of current sequences with computed fitness
        self.singles={}
        self.ruggedness=ruggedness
        self.debug=False
        self.wt_seq=wt_full_seq
        self.kill_list={}
        self.kill_switch=kill_switch
        self.eta=eta

    def set_debug(self,debug):
        self.debug=debug

    """For additive models, you need some data (possibly hypothetical) of singles fitnesses to generate multi-mutant fitnesses"""    
    def populate_singles(self,singles_data):
        for mutation in singles_data:
            self.singles[mutation]=singles_data[mutation]

    def _fitness_function(self,sequence):
        additive=self.compute_fitness_additive(sequence)
        if self.ruggedness:
            noise= self.ruggedness*(np.random.normal(scale=self.eta))
        else:
            noise=0

        return (1-self.ruggedness)*additive+self.ruggedness*noise 

    def populate_model(self,data=None):
        for sequence in data:
            self.sequences[sequence]=self._fitness_function(sequence)

    def get_fitness(self,sequence):
         if sequence in self.sequences:
            return self.sequences[sequence]
         else:
            if self.kill_switch and self.should_be_killed(sequence):
               self.sequences[sequence]=self._fitness_function(sequence)-10
            else:
               self.sequences[sequence]=self._fitness_function(sequence)
            return self.sequences[sequence]   
        
    def compute_fitness_additive(self,sequence_mask):
        if self.singles=={}:
           print ("singles data not available")  
        mask_singles=break_down_sequence_to_singles(sequence_mask)
        fitness=0
        singles=[translate_mask_to_full_seq(seq,self.wt_seq) for seq in mask_singles]
        for mutation in singles:
            if self.debug:
                print (mutation,self.singles[mutation])
            fitness+=self.singles[mutation]
        return fitness

    def should_be_killed(self,sequence):
        #return False
        len_of_motif=max(1,int(self.kill_switch*len(sequence)))
        if "E"*len_of_motif not in sequence: 
            return True
        return False



    def __str__(self):
        return "Simple_Additive"

class NL_Additive(Simple_Additive):
    """ This can be used for instance in packaging/GFP models, it accepts a function as a parameter, that transforms the additive
    underlying model: e.g. sigmoid, I-spline (e.g. Plotkin group paper 2018),..."""
    def __init__(self,wt_seq,NL_function,ruggedness=0.0,kill_switch=0.1):
        self.sequences={} #cache of currently known sequences
        self.singles={}
        self.wt_seq=wt_seq
        self.ruggedness=ruggedness
        self.NL_function=NL_function
        self.debug=False
        self.kill_switch=kill_switch

    def _fitness_function(self,sequence):
        return self.NL_function((self.compute_fitness_additive(sequence)))+(self.ruggedness*(np.random.standard_normal()))


    #add additive max and min compute 


    def __str__(self):
        return "NL_additive"


class RMF(Simple_Additive):
    def __init__(self,wt_full_seq, c, eta_std,kill_switch=0.0,alphabet=AAS):

        self.sequences={} #cache of current sequences with computed fitness
        self.singles={}
        self.c=c
        self.debug=False
        self.wt_seq=wt_full_seq
        self.eta_std=eta_std
        self.kill_list={}
        self.kill_switch=kill_switch
        self.alphabet=alphabet
        self.maximum_additive=""
        self.theta=self.c/(self.eta_std)

    def compute_maximum_additive(self):
        top_seq=[]
        for pos in range(len(self.wt_seq)):
            current_max=-1000
            current_aa="X"
            for aa in self.alphabet:
                mutant=self.wt_seq[:pos]+aa+self.wt_seq[pos+1:] 
                mutant_fitness=self.singles[mutant]
                if mutant_fitness>current_max:
                    current_max=mutant_fitness
                    current_aa=aa
            top_seq.append(current_aa)

        self.maximum_additive="".join(top_seq)


    def _fitness_function(self,sequence):

        distance=editdistance.eval(sequence,self.maximum_additive)
        fitness=-self.c*distance+np.random.normal(scale=self.eta_std)
        return fitness

         

class Epistatic(Simple_Additive):
    """This is a more general implementation of the NK model, the 'N,K' model can be achieved
     by passing an N,K interaction matrix. 
     See utils.interaction_map_generators for helper functions to make NK or other interaction matrices"""

    def __init__(self,interaction_map,aa_map,wt_full_seq,kill_switch=0.0,ruggedness=0.0):
        self.sequences={} #cache of current sequences with computed fitness
        self.singles={}
        self.interaction_map=interaction_map #interaction map between positions. This is an N x N matrix. 
        self.aa_map=aa_map # aa affinities to each other 20 x 20  (this cannot be position dependent for this model)
        self.wt_seq=wt_full_seq #wt full sequence (not just the mask)
        self.ruggedness=ruggedness
        self.debug=False
        self.kill_switch=kill_switch


    def _fitness_function(self,sequence):
        #print (sequence,self.wt_seq)
        full_seq=translate_mask_to_full_seq(sequence,self.wt_seq)
        #print (full_seq)
        single_masks=break_down_sequence_to_singles(full_seq)

        single_masks=[translate_mask_to_full_seq(seq,self.wt_seq) for seq in single_masks]

        #print (single_masks)
        fitness=0
        for i in range(len(full_seq)):
            for j in range(len(full_seq)):
               # print (single_masks[i],single_masks[j])
                position_interaction=self.interaction_map[i][j]*((self.singles[single_masks[i]]+self.singles[single_masks[j]])/2)
                if i!=j:
                   aa_i=full_seq[i]
                   aa_j=full_seq[j]
                   aa_i_index=translate_aa_to_index(aa_i)
                   aa_j_index=translate_aa_to_index(aa_j)
                   aa_interaction= self.aa_map[aa_i_index][aa_j_index]
                else:
                   aa_interaction=1 
                
                fitness+=(position_interaction*aa_interaction)#/len(full_seq)

                if self.debug :
                    if self.interaction_map[i][j]!=0:
                         print (i,j,position_interaction,aa_interaction,fitness)


        return fitness+(self.ruggedness*(np.random.standard_normal()))

    def __str__(self):
        return "Epistatic"


class Classic_NK(Simple_Additive):
    def __init__(self,interaction_map,kill_switch=0.0):
        self.sequences={} #cache of currently known sequences
        self.singles={}
        self.interaction_map,self.N,self.K=interaction_map
        self.epis_table=self.populate_epis_table()
        self.kill_switch=kill_switch

    def populate_epis_table(self):
        epis_table={}
        for i,row in enumerate(self.interaction_map):
            epis_table[i]={}
            interactions=np.nonzero(row)[0]
            NK_table=generate_all_binary_combos(len(interactions)-1)  
            for variant in NK_table:
                epis_table[i]["".join(variant)]=np.random.uniform()
            epis_table[i]["interactors"]=tuple(interactions)
        return epis_table

    def compute_fitness_additive(self,sequence):
        z=0
        current_fitness=0
        for i,s in enumerate(sequence):
            if s!="0":
               z+=1
               mutation="0"*i+"1"+"0"*(len(sequence)-(i+1))
               current_fitness+=self.get_fitness(mutation)
        if z==0:
            return self.get_fitness(sequence)

        return current_fitness*1./z
                



    def _fitness_function(self,sequence):
        fitness=0
        for i in range(len(sequence)):
            interactions=self.epis_table[i]["interactors"]
            key=[]
            for j in interactions:
                key.append(sequence[j])

            strkey="".join(key)
            #print(strkey)
           # print (strkey,self.epis_table[i][strkey])
            fitness+=self.epis_table[i][strkey]

        return fitness/len(sequence)

    def __str__(self):
        return "NK_classic"




