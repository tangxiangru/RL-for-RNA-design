from utils.RL_utils import *
from meta.explorer import Explorer

from utils.sequence_utils import translate_string_to_one_hot,translate_one_hot_to_string, generate_random_mutant
import numpy as np
import random
import editdistance

from utils.adaptive_sampling import *



class CE(Explorer):

    def __init__(self,model, initial_sequences, wt, alphabet="01",batch_size=1,virtual_screen=1,temperature=0.1,recomb_rate=0):
        self.model=model
        self.wt=wt

        self.alphabet=alphabet

        self.batch_size=batch_size

        self.sequences=[seq for seq in initial_sequences]
        self.size=len(self.sequences)

        self.best_seqs=[]
        self.best_seqs_len=1000
        self.temperature=temperature
        self.batch=[]
        self.recomb_rate=recomb_rate

        self.virtual_screen=virtual_screen
        self.top_sequence=[sorted([(self.model.get_fitness(seq),seq,self.model.cost) for seq in self.sequences])[-1]]


    def update_belief_state(self):
        pass


    def generate_sequences(self):
        offspring=[]
        for seq in self.sequences:
            cloud=[]

            c=list(set([generate_random_mutant(seq,((1/(2*len(seq)))*1.1**i)/len(seq),alphabet=self.alphabet) for i in range(self.virtual_screen)]))
            for ic in c:
                if ic not in self.model.measured_sequences:
                    cloud.append(ic)
                    cloud=list(set(cloud)) 
            count=0
            while len(set(cloud))<self.virtual_screen and count<100:
                c=list(set([generate_random_mutant(seq,((1/(1*len(seq)))*1.5**i)/len(seq),alphabet=self.alphabet) for i in range(self.virtual_screen)]))
                for ic in c:
                    if ic not in self.model.measured_sequences:
                        cloud.append(ic)
                        cloud=list(set(cloud))  
                count+=1 
                #if count>10:
                 #   print("stuck here",len(cloud))

            cloud=list(set(cloud))[:self.virtual_screen]

            offspring.extend(cloud)

        seq_and_fitness=[]
        for seq in set(offspring):
            seq_and_fitness.append((self.model.get_fitness(seq),seq))
        return  sorted(seq_and_fitness,reverse=True)            
        #note, you can have VS =100 and still take the top 10 or 20% right? that is the batch size and VS don't need to be coupled.

    def pick_action(self):
        new_seqs_and_fitnesses=self.generate_sequences()

        new_batch=new_seqs_and_fitnesses[:self.batch_size]
        batch_seq=[]


        for fit,seq in new_batch:

            batch_seq.append(seq)
            self.model.measure_true_landscape([seq]) #can't uncomment this if you want top seq to be accurate
            fitness=self.model.get_fitness(seq)
            if fitness>self.top_sequence[-1][0]:
                  self.top_sequence.append((fitness,seq,self.model.cost))


        self.sequences=batch_seq
        return batch_seq

#TODO add adaptive sampling variant
class AdaCE(CE):

    def pick_action(self):
        new_seqs_and_fitnesses=self.generate_sequences()

        raw_batch=new_seqs_and_fitnesses[:self.batch_size*2]
        raw_batch_seq=[s[1] for s in raw_batch]

        diverse_batch=amortized_filtering(raw_batch_seq,5,self.batch_size,lambda x: get_sum_entropy(x,alphabet=self.alphabet),OPT=len(raw_batch_seq[0]),eps=0.1)
        new_batch=diverse_batch[:self.batch_size]

        batch_seq=[]


        for seq in new_batch:

            batch_seq.append(seq)
            self.model.measure_true_landscape([seq]) #can't uncomment this if you want top seq to be accurate
            fitness=self.model.get_fitness(seq)
            if fitness>self.top_sequence[-1][0]:
                  self.top_sequence.append((fitness,seq,self.model.cost))


        self.sequences=batch_seq
        return batch_seq





