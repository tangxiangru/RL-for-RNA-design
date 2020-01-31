import numpy as np 
import random 

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())-1

"""This is a method to sample k integers from a list of size N uniformly"""
def reservoir_sample(iterable, k):
    reservoir = []
    for t, item in enumerate(iterable):
        if t < k:
            reservoir.append(item)
        else:
            m = np.random.randint(0,t+1)
            if m < k:
                reservoir[m] = item
    return sorted(reservoir)

def generate_interaction_map(N,density):
    """generates an interaction map between positions, where the density 
    controls the amount of interaction in the sequence (symmetric)"""
    interaction_map=np.eye(N,N)
    for i in range(N-1):
        for j in range(i,N):
            if i!=j:
               if np.random.uniform()<density:
                  interaction_map[i][j]=1
    interaction_map=symmetrize(interaction_map)
    return interaction_map

def generate_NK_map(N,K):
    """generates a random NK interaction map between positions
    K determines how many other positions affect the fitness contribution from this position
    NOTE: this is NOT symmetric"""

    interaction_map=np.eye(N,N)
    for pos in range(N):
        indexes=reservoir_sample(range(N-1),K) #generate k random indices uniformly
        for i in indexes:
            if i<pos:
                interaction_map[pos][i]=1
            else:
                interaction_map[pos][i+1]=1
    return (interaction_map,N,K)

def generate_random_aa_interaction_map(density):
    interaction_map=np.ones((20,20))
    for i in range(19):
        for j in range(i,20):
            if i!=j:
               if np.random.uniform()<density:
                  interaction_map[i][j]=1-np.random.normal()
    
    interaction_map=symmetrize(interaction_map)
    
    for i in range(20):
        if np.random.uniform()<density:
            interaction_map[i][i]=1-np.random.normal()
        else:
            interaction_map[i][i]=1

    return interaction_map

def generate_random_aa_interaction_map_discrete(no_epis):
    interaction_map=np.ones((20,20))
    for i in range(19):
        for j in range(i,20):
            if i!=j:
                x=np.random.uniform()
                if x<no_epis:
                    interaction_map[i][j]=1
                else:
                    interaction_map[i][j]=np.random.uniform(-1,2)
    
    interaction_map=symmetrize(interaction_map)
    
    for i in range(20):
        if np.random.uniform()<no_epis:
            interaction_map[i][i]=np.random.uniform(-1,2)
        else:
            interaction_map[i][i]=1

    return interaction_map



