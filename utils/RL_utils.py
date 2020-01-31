import numpy as np
import random
import bisect
from utils.sequence_utils import translate_string_to_one_hot


def renormalize_moves(one_hot_input,rewards_output):
    """ensures that staying in place gives no reward"""
    zero_current_state=(one_hot_input-1)*-1
    return np.multiply(rewards_output,zero_current_state)

def walk_away_renormalize_moves(one_hot_input, one_hot_wt, rewards_output):
    """ensures that moving toward wt is also not useful"""
    zero_current_state=(one_hot_input-1)*-1
    zero_wt=((one_hot_wt-1)*-1)
    zero_conservative_moves=np.multiply(zero_wt,zero_current_state)
    return np.multiply(rewards_output,zero_conservative_moves)

def get_all_singles_fitness(model,sequence,alphabet):
    prob_singles=np.zeros((len(alphabet),len(sequence)))
    for i in range(len(sequence)):
        for j in range(len(alphabet)):
            putative_seq=sequence[:i]+alphabet[j]+sequence[i+1:]
           # print (putative_seq)
            prob_singles[j][i]=model.get_fitness(putative_seq)
    return prob_singles
 
def sample_boltzmann(matrix):
    i,j=matrix.shape
    flat_mat=np.exp (matrix.flatten())
    cumsum_flat_mat=np.cumsum(flat_mat)
    max_val=max(cumsum_flat_mat)
    # print (flat_mat)

    #print (cumsum_flat_mat)
    #print (max_val)

    sample=np.random.uniform(0,max_val)
    index=bisect.bisect_left(cumsum_flat_mat,sample)  
    x,y=(int(index/j),index%j)
    #  print(index)
    output=np.zeros((i,j))
    # print (x,y)
    output[x][y]=matrix[x][y]
    return output

def sample_gumbel(matrix):
    i,j=matrix.shape
    flat_mat=matrix.flatten()+np.random.gumbel(size=i*j)
    cumsum_flat_mat=np.cumsum(flat_mat)
    max_val=max(cumsum_flat_mat)
    # print (flat_mat)

    #print (cumsum_flat_mat)
    print (max_val)

    sample=np.random.uniform(0,max_val)
    index=bisect.bisect_left(cumsum_flat_mat,sample)  
    x,y=(int(index/j),index%j)
    #  print(index)
    output=np.zeros((i,j))
    # print (x,y)
    output[x][y]=matrix[x][y]
    return output

def sample_thompson(matrix):
    i,j=matrix.shape
    flat=np.exp(matrix.flatten())
    flat /= np.sum(flat)
    index = np.random.choice([i for i in range(len(flat))],p=flat)
    x,y =(int(index/j),index%j)
    output = np.zeros((i,j))
    output[x][y] = matrix[x][y]
    return output

def sample_greedy(matrix):
    i,j=matrix.shape
    max_arg=np.argmax(matrix)
    y=max_arg%j
    x=int(max_arg/j)
    output=np.zeros((i,j))
    output[x][y]=matrix[x][y]
    return output

# takes top n greedy
def sample_n_greedy(matrix, n=2):
    i,j=matrix.shape
    max_n_arg=np.argpartition(matrix.flatten(), -n)[-n:]
    outputs = []
    for max_arg in max_n_arg:
        y=max_arg%j
        x=int(max_arg/j)
        output=np.zeros((i,j))
        output[x][y]=matrix[x][y]
        outputs.append(output)
    return outputs

def sample_random(matrix):
    i,j=matrix.shape
    non_zero_moves=np.nonzero(matrix)
   # print (non_zero_moves)
    k=len(non_zero_moves)
    l=len(non_zero_moves[0])
    if k!=0 and l!=0:
        rand_arg=random.choice([[non_zero_moves[alph][pos] for alph in range(k)] for pos in range(l)])
    else:
        rand_arg=[random.randint(0,i-1),random.randint(0,j-1)]
    #print (rand_arg)
    y=rand_arg[1]
    x=rand_arg[0]
    output=np.zeros((i,j))
    output[x][y]=matrix[x][y]
    return output   
    
def construct_mutant_from_sample(pwm_sample,one_hot_base):
    one_hot=np.zeros(one_hot_base.shape)
    one_hot+=one_hot_base
    i,j=np.nonzero(pwm_sample)# this can be problematic for non-positive fitnesses
    one_hot[:,j]=0
    one_hot[i,j]=1
    return one_hot

def make_one_hot_train_test(genotypes,model,alphabet,split=0.0):
    genotypes_one_hot=[translate_string_to_one_hot(genotype,alphabet) for genotype in genotypes]
    genotype_fitnesses=[get_all_singles_fitness(model,genotype,alphabet) for genotype in genotypes]
    train_x=[]
    test_x=[]
    train_y=[]
    test_y=[]
    if split>0: #check this to avoid calling the randomizer if not required
        for x,y in zip(genotypes_one_hot,genotype_fitnesses):
                if random.random()<split:
                   test_x.append(x)
                   test_y.append(y)
                else:
                    train_x.append(x)
                    train_y.append(y)
        train_x=np.stack(train_x)
        train_y=np.stack(train_y)
        test_y=np.stack(test_y)
        test_x=np.stack(test_x)
        return train_x,train_y,test_x,test_y
    else:
        train_x=np.stack(genotypes_one_hot)
        train_y=np.stack(genotype_fitnesses)

        return train_x,train_y,test_x,test_y