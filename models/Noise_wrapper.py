import editdistance
import numpy as np
from meta.model import Model
import random
from utils.RL_utils import translate_string_to_one_hot
from sklearn.metrics import explained_variance_score, r2_score
#from sklearn import cross_validation
import keras

class Noise_wrapper(Model):
    def __init__(self,model,scale=1,noise_bias=0,noise_min=-1,noise_max=1,noise_alpha=0.5,always_costly=False,natural_mode=False,local_mode=False):

        self.model_sequences={}
        self.measured_sequences={}
        self.model=model
        self.cost=0
        self.scale=scale
        self.noise_bias=noise_bias
        self.noise_min=noise_min
        self.noise_max=noise_max
        self.DF=[]
        self.noise_alpha=noise_alpha
        self.costly=always_costly
        self.natural_mode=natural_mode
        self.local_mode=local_mode
        self.wt=""
        self.one_hot_sequences={}
        self.alphabet=""
        self.evals = 0

    def populate_model(self,data=None):
        pass

    def bootstrap(self,wt,alphabet):
        sequences=[wt]
        self.wt=wt
        self.alphabet=alphabet
        for i in range(len(wt)):
            tmp=list(wt)
            for j in range(len(alphabet)):
                tmp[i]=alphabet[j]
                sequences.append("".join(tmp))
        self.measure_true_landscape(sequences)
        self.DF=[self.get_fitness(seq) for seq in sequences]

    def reset(self):
        self.model_sequences={}
        self.measured_sequences={}
        self.cost=0
        self.evals = 0


    def get_min_distance(self,sequence):
        new_dist=1000
        for seq in self.measured_sequences:
            dist=editdistance.eval(sequence,seq)
            if dist==1:
               new_dist=1
               break
            else:
                new_dist=min(dist,new_dist)
        return new_dist

    def add_noise(self,sequence,distance):
        signal=self.model.get_fitness(sequence)
        noise=random.choice(self.DF)
        alpha=(self.noise_alpha)**distance
        #print (signal,noise, alpha)

        #for the noise model, get the distribution of known fitnesses, then take the weighted average of real
        # signal and a sample from that distribution weighted like fit_eff*(1-(1/2)^distance)+signal*(1/2)^ditance
        return signal,noise,alpha

    def _fitness_function(self,sequence):

        if self.noise_alpha<1 and not self.natural_mode:

            distance=self.get_min_distance(sequence)

            signal,noise,alpha=self.add_noise(sequence,distance)
            surrogate_fitness=signal*alpha+noise*(1-alpha)
        elif self.local_mode:
            surrogate_fitness=self.model.get_fitness(sequence)*(0.99)**editdistance.eval(self.wt,sequence)
        else:
            surrogate_fitness=self.model.get_fitness(sequence)
        #print (surrogate_fitness)
        return surrogate_fitness

    def measure_true_landscape(self,sequences):
        for sequence in sequences:
            if sequence not in self.measured_sequences:
                    self.cost+=1
                    self.measured_sequences[sequence]=self.model.get_fitness(sequence)
        self.model_sequences={} #think about this carefully, if this is your "update" then you might classify the same point as noisy in one round and not noisy in the next round, even though your measure didn't give any extra info
        self.DF=[self.measured_sequences[seq] for seq in self.measured_sequences]

    def get_fitness(self,sequence):
        if sequence in self.measured_sequences:
            return self.measured_sequences[sequence]
        elif sequence in self.model_sequences:
            return self.model_sequences[sequence]
        else:
            self.model_sequences[sequence]=self._fitness_function(sequence)
            self.evals += 1
            if self.costly:
               self.cost+=1 
            return self.model_sequences[sequence]   


class estimate_additive_from_wt(Noise_wrapper):

    def _fitness_function(self,sequence):
        if self.natural_mode:
            fitness=self.model.get_fitness(sequence)
        else:
            fitness=self.model.compute_fitness_additive(sequence)
        return fitness


class Gaussian_noise_landscape(Noise_wrapper):
     
      def _fitness_function(self,sequence):
        if self.natural_mode:
            fitness=self.model.get_fitness(sequence)
        else:
            fitness=self.model.get_fitness(sequence)+np.random.normal(scale=self.noise_alpha)
        return fitness

class DF_noise_landscape(Noise_wrapper):
      
      def _fitness_function(self,sequence):
        if self.natural_mode:
            fitness=self.model.get_fitness(sequence)
        else:
            fitness=self.noise_alpha*self.model.get_fitness(sequence)+(1-self.noise_alpha)*random.choice(self.DF)
        return fitness



class skModel(Noise_wrapper):

    def assign_skmodel(self,model,batch_update=False,alphabet="UCGA"):
        self.batch_update=batch_update
        self.skmodel=model
        self.noise_alpha=0.25
        self.alphabet=alphabet


    def reset(self):
        self.model_sequences={}
        self.measured_sequences={}
        self.cost=0
        self.skmodel=sklearn.base.clone(self.skmodel)
        
    def bootstrap(self,wt,alphabet):
        sequences=[wt]
        self.wt=wt
        self.alphabet=alphabet
        for i in range(len(wt)):
            tmp=list(wt)
            for j in range(len(alphabet)):
                tmp[i]=alphabet[j]
                sequences.append("".join(tmp))
        self.measure_true_landscape(sequences)
        self.one_hot_sequences={sequence:(translate_string_to_one_hot(sequence,self.alphabet).flatten(),self.measured_sequences[sequence]) for sequence in sequences} #removed flatten for nn
        self.update_model(sequences)

    def resetNaive(self):
        self.model_sequences={}
        self.measured_sequences={}
        self.cost=0

    def reset(self, model):
        self.skmodel=model
        self.noise_alpha=0.25
        self.model_sequences={}
        self.measured_sequences={}
        self.cost = 0
        self.evals = 0

    def update_model(self,sequences):
        X=[]
        Y=[]
        for sequence in sequences:
            if sequence not in self.one_hot_sequences:# or self.batch_update:
                #print (sequence)
                x=translate_string_to_one_hot(sequence,self.alphabet).flatten()
                y=self.measured_sequences[sequence]
                self.one_hot_sequences[sequence]=(x,y)
                X.append(x)
                Y.append(y)
            else:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)
        X=np.array(X)
        Y=np.array(Y)

        self.retrain_model()
        try:
            y_pred=self.skmodel.predict(X)
            self.noise_alpha=explained_variance_score(Y,y_pred)
            self.noise_alpha=r2_score(Y,y_pred)
        except:
            pass

        self.skmodel.fit(X,Y)


    def retrain_model(self):
        X,Y=[],[]
        random_sequences=random.sample(self.one_hot_sequences.keys(),min(len(self.one_hot_sequences.keys()),500))
        for sequence in random_sequences:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)

        X=np.array(X)
        Y=np.array(Y)
        self.skmodel.fit(X,Y)

    def _fitness_function(self,sequence):
        x=np.array([translate_string_to_one_hot(sequence,self.alphabet).flatten()])

        return max(min(200, self.skmodel.predict(x)[0]),-200)


class nnModel(Noise_wrapper):

    def assign_skmodel(self,model,batch_update=False,epochs=20,batch_size=10,validation_split=0.1,alphabet="UCGA"):
        self.batch_update=batch_update
        self.skmodel=model
        self.batch_size=batch_size
        self.validation_split=validation_split
        self.epochs=epochs
        self.noise_alpha=0.25
        self.alphabet=alphabet


    def reset(self):
        self.model_sequences={}
        self.measured_sequences={}
        self.cost=0
        self.skmodel=keras.models.clone_model(self.skmodel)
        self.skmodel.compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])

    def reset(self, model):
        self.skmodel=model
        self.noise_alpha=0.25
        self.model_sequences={}
        self.measured_sequences={}
        self.cost=0

    def bootstrap(self,wt,alphabet):
        sequences=[wt]
        self.wt=wt
        self.alphabet=alphabet
        for i in range(len(wt)):
            tmp=list(wt)
            for j in range(len(alphabet)):
                tmp[i]=alphabet[j]
                sequences.append("".join(tmp))
        self.measure_true_landscape(sequences)
        self.one_hot_sequences={sequence:(translate_string_to_one_hot(sequence,self.alphabet),self.measured_sequences[sequence]) for sequence in sequences} #removed flatten for nn
        self.update_model(sequences)



    def update_model(self,sequences):
        X=[]
        Y=[]
        for sequence in sequences:
            if sequence not in self.one_hot_sequences:# or self.batch_update:
                x=translate_string_to_one_hot(sequence,self.alphabet)#.flatten()
                y=self.measured_sequences[sequence]
                self.one_hot_sequences[sequence]=(x,y)
                X.append(x)
                Y.append(y)
            else:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)
        X=np.array(X)
        Y=np.array(Y)

        try:
            y_pred=self.skmodel.predict(X)
          #  self.noise_alpha=explained_variance_score(Y,y_pred)
            self.noise_alpha=r2_score(Y,y_pred)

        except:
            pass

        self.skmodel.fit(X,Y,epochs=self.epochs,validation_split=self.validation_split,batch_size=self.batch_size,verbose=0)
        if not self.batch_update:
            self.retrain_model()

    def retrain_model(self):
        X,Y=[],[]
        random_sequences=random.sample(self.one_hot_sequences.keys(),min(len(self.one_hot_sequences.keys()),500))
        for sequence in random_sequences:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)

        X=np.array(X)
        Y=np.array(Y)

        self.skmodel.fit(X,Y,epochs=self.epochs,validation_split=self.validation_split,batch_size=self.batch_size,verbose=0)


    def _fitness_function(self,sequence):
        try:
            x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]
        except:
            print (sequence)
        return max(min(200, self.skmodel.predict(x)[0][0]),-200)



class MultiModel(Noise_wrapper):

        def assign_models(self,list_of_models_dict):
            self.list_of_models_dict=list_of_models_dict
            #[{'model:<model_object','epochs':epochs,'batch_size':batch_size},{}]
            self.model_performances=[0 for i in range(len(self.list_of_models_dict))]
            self.best_model=self.list_of_models_dict[0]['model']
        def reset(self):
            self.measured_sequences={}
            self.cost=0
            self.model_performances=[0 for i in range(len(self.list_of_models_dict))]

            for model_dict in self.list_of_models_dict:
                model_dict['model']=keras.models.clone_model(model_dict['model'])
                model_dict['model'].compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])
       
        def bootstrap(self,wt,alphabet):
            sequences=[wt]
            self.wt=wt
            self.alphabet=alphabet
            for i in range(len(wt)):
                tmp=list(wt)
                for j in range(len(alphabet)):
                    tmp[i]=alphabet[j]
                    sequences.append("".join(tmp))
            self.measure_true_landscape(sequences)
            self.one_hot_sequences={sequence:(translate_string_to_one_hot(sequence,self.alphabet),self.measured_sequences[sequence]) for sequence in sequences} #removed flatten for nn
            self.update_model(sequences)


        def update_model(self,sequences, bootstrap=True):
            X=[]
            Y=[]
            for sequence in sequences:
                if sequence not in self.one_hot_sequences:# or self.batch_update:
                    x=translate_string_to_one_hot(sequence,self.alphabet)#.flatten()
                    y=self.measured_sequences[sequence]
                    self.one_hot_sequences[sequence]=(x,y)
                    X.append(x)
                    Y.append(y)
                else:
                    x,y=self.one_hot_sequences[sequence]
                    X.append(x)
                    Y.append(y)
            X=np.array(X)
            Y=np.array(Y)

            for i,model_dict in enumerate(self.list_of_models_dict):
                try:

                    y_pred=model_dict['model'].predict(X)
              #  self.noise_alpha=explained_variance_score(Y,y_pred)
                    self.model_performances[i]=r2_score(Y,y_pred)

                except:
                    pass
                indices=[]
                for k in range(int(len(X)/2)):
                    indices.append(random.randint(0,len(X)-1))
                model_dict['model'].fit(np.take(X,indices,axis=0),np.take(Y,indices,axis=0),epochs=model_dict['epochs'],validation_split=0,batch_size=model_dict['batch_size'],verbose=0)
            best_model_index=np.argmax(self.model_performances)
            self.best_model=self.list_of_models_dict[best_model_index]['model']


        def _fitness_function(self,sequence):
            x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]

            return max(min(200, self.best_model.predict(x)[0][0]),-200)

class MultiModelDist(MultiModel):
        def get_uncertainty(self,sequence):
            x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]

            x_predicts=[m["model"].predict(x)[0][0] for m in self.list_of_models_dict]

            return np.mean(x_predicts), np.std(x_predicts), x_predicts

        def _fitness_function(self,sequence):
            x_mean,x_std, all_x=self.get_uncertainty(sequence)

            return max(min(200, self.best_model.predict(x)[0][0]),-200)


