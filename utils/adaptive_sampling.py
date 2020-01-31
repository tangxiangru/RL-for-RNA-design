
import random
import numpy as np
from scipy.stats import entropy
def f_set(func,S,X):
    set_S=set(S)
    set_X=set(X)
    un=list(set_S.union(set_X))
    return func(un)-func(S)

def E(func, S, X,num):
    #print (len(S), len(X),num)
    Rs=sum([f_set(func,list(S),random.sample(X,min(num,len(X))))/10 for i in range(10)])
    return Rs

def E1(func, S,X, a,num):
    #print (len(S), len(X),num)
    Rs=sum([f_set(func,list(set(S).union(set(random.sample(X,min(num,len(X)))))-set(a)),a)/10 for i in range(10)])
    return Rs


def filter_f(N,S,r,k,func,OPT=25,eps=0.01):
    X=N[:]
    e=eps
    RHS=(1-e)*(OPT-func(S))/r
    while E(func,S, X, int(k/r)) <RHS:
        to_remove=[]
        for a in X:
            sS=set(S)
            if E1(func, S,X,[a],int(k/r))<((1+e/2)*RHS/k):
                to_remove.append(a)
        
        for a in to_remove:
            X.remove(a)
    return X

def iterative_filter(N,r,k,func):
    S=[]
    for i in range(r):
        X=filter_f(N,S,r,k,func)
        sS=set(S)
        S=list(sS.union(set(random.sample(X,min(int(k/r),len(X))))))
        
    return S

def amortized_filtering(N,r,k,func,OPT=25,eps=0.1):
    S=set()
    eps=eps
    for i in range(int(20/eps)):
        X=N[:]
        T=set()
        while f_set(func,S,T)< (int(20/eps)*(OPT-func(S))/r) and len(set(S).union(T))<k:
            X=filter_f(N,S,r,k,func,OPT=OPT,eps=eps)
            T=T.union(set(random.sample(X,int(k/r))))
        S=list(set(S).union(T))
    return list(S)
        

def get_sum_entropy(sequences,alphabet="01"):
    if len(sequences)==0:
        return 0
    seqs_int_array=[]
    for seq in sequences:
        row=[i for i in list(seq)]
        seqs_int_array.append(row)
    count_matrix=np.zeros((len(alphabet),len(sequences[0])))
    seqs_int_array=np.array(seqs_int_array)
    for j in range(len(seq)):
        for i in range(len(alphabet)):
            count_matrix[i][j]=list(seqs_int_array[:,j]).count(alphabet[i])
    
    return sum(entropy(count_matrix,base=2))