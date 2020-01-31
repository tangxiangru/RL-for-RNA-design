
import itertools
import editdistance

AAS="ILVAGMFYWEDQNHCRKSTP" #list of Amino-Acids


def compute_sequence_peakiness(model,sequence,alphabet=AAS):
    higher_than_neighbor=0
    neighbor=[s for s in sequence]
    better_neighbors=[]
    for position in range(len(sequence)):
        for aa in alphabet:
            if aa!=sequence[position]:
               neighbor[position]=aa
               neighbor_string="".join(neighbor)
               if model.get_fitness(sequence)>model.get_fitness(neighbor_string):
                  higher_than_neighbor+=1
               else:
                  better_neighbors.append((neighbor_string,model.get_fitness(neighbor_string)))
               neighbor=[s for s in sequence]

    higher_than_neighbor_prop=higher_than_neighbor/(19*len(sequence))
    return better_neighbors,higher_than_neighbor_prop

def is_sequence_a_peak(model,sequence,alphabet=AAS):
    neighbor=[s for s in sequence]
    for position in range(len(sequence)):
        for aa in alphabet:
            if aa!=sequence[position]:
               neighbor[position]=aa
               neighbor_string="".join(neighbor)
               if model.get_fitness(sequence)<model.get_fitness(neighbor_string):
                  return False
               neighbor=[s for s in sequence]
    return True

def brute_force_close_peaks(model, wt, depth, alphabet=AAS):
    N = len(wt)
    order = [''.join(i) for i in itertools.product(alphabet, repeat = depth)]
    changeIndices = itertools.combinations(range(N), depth)

    # generate possible sequences
    all_seq = set()
    for change in changeIndices:
        for mut_seq in order:
            new_seq = list(wt)
            for k, ind in enumerate(change):
                new_seq[ind] = mut_seq[k]
            all_seq.add(''.join(new_seq))

    # finds peaks
    peaks=[]
    for seq in all_seq:
       if is_sequence_a_peak(model,seq,alphabet):
          peaks.append(seq)
    return peaks

def brute_force_all_peaks(model,length,alphabet=AAS):
   all_sequences = [''.join(i) for i in itertools.product(alphabet, repeat = length)]
   peaks=[]
   for seq in all_sequences:
       if is_sequence_a_peak(model,seq,alphabet):
          peaks.append(seq)
   return peaks  

def peaks_found(s_peaks,seq_set):
    out=[]
    seqs=[seq[1] for seq in seq_set]
   # print (seqs)
    for peak in s_peaks:
        #print (peak[1])
        if peak[1] in seqs:
           out.append(peak)
        
    return out

def distance_to_best_peak(s_peaks,seq_set):
    out=[]
    s_peaks=[s_peak for s_peak in s_peaks]
    for seq in seq_set:
        out.append(editdistance.eval(s_peaks[-1][1],seq[1]))
        
    return sorted(out)


def get_real_fitness_for_best_seqs(landscape,best_seqs):
    seqs=[]
    for seq in best_seqs:
        seqs.append(landscape.get_fitness(seq[1]))
    return sorted(seqs)


def get_distance_fitness(wt,landscape):
    output=[]
    for seq in landscape.measured_sequences:
        output.append((editdistance.eval(wt,seq),landscape.get_fitness(seq)))
    return output
