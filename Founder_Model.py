import numpy as np
import copy

################################### Parameters #############################################

N = 300                                               # population size
Nbud = 250                                            # budding size
seed = 50                                             # seed size
r = 0.2                                               # growth rate
k = 50                                                # number of village
mu = 0.0249                                           # mutation rate
m = 0.135                                             # migration rate
p = 0.1                                               # 1 - heritability
sigma = 0.8                                           # reproductive advantage
delta = 0.00                                          # fraction of dominance (set to zero for nondominance island model)
nd = int(round(N*delta))                              # number of dominant individuals
ndseed = int(round(seed*delta))

T = 100                                               # number of generation
ns = 60                                               # sampling size
Nens = 200                                            # number of realizations

##############################################################################################
''' The origin (where the first village is drawn from) is simulated with neutral model '''

populationO = np.ones(N, dtype=np.int)          # single haplotype
probO = np.ones(N, dtype=float)                 # reproduction probability
probO = probO / float(N)                        # normalized the sum to one
t = 0
pop0 = copy.copy(populationO)                   # current generation
pop1 = copy.copy(populationO)                   # next generation
HO = 1                                          # number which had been used to represent the haplo
while t < T:
    numoffspringO = np.random.multinomial(N, probO, size=1)
    # produce N number of offsprings, prob0: prob for each (of N) father to be chosen, size: do the drawing 1 time
    zS = 0                                                     # record index of son
    for ii in range(N):                                        # go over all father
        if numoffspringO[0,ii] > 0:                            # if father ii reproduce
            sonO = np.arange(zS, zS + numoffspringO[0, ii])    # assign index to the son
            pop1[sonO] = pop0[ii]                              # son's haplotype follow father
            zS = zS + numoffspringO[0, ii]

    mup = np.random.random((N, 1))                             # mutation probability
    mutationO = [x for x in xrange(N) if mup[x, 0] < mu]
    if len(mutationO) >= 1:
        pop1[mutationO] = np.arange(1, len(mutationO)+1) + HO  # create new haplotypes
        HO += len(mutationO)
    pop0 = copy.copy(pop1)
    t += 1

US, countsS = np.unique(pop0, return_counts=True)
print len(US)
##############################################################################################

HetS = np.zeros((Nens, k-5), dtype=np.float)                    # heterozygosity of haplotype

############################### Founder Model ###############################

for ens in range(Nens):
    print ens
    done = False
    ########################## Initialization #############################

    G0S = np.zeros((N,k), dtype=np.int)               # current generation
    G0S[0:seed, 0] = np.random.choice(pop0, seed, replace=False)          # first village
    G1S = copy.copy(G0S)                              # updated generation
    HS = np.max(pop0)                                 # max number used to represent the haplotype
    M = np.zeros(k, dtype=np.int)                     # population size of each village
    M[0] = seed                                       # population size of the first village
    nv = 1                                            # number of active village
    probS = np.zeros((N,k), dtype=float)              # probability to reproduce
    probS[0:seed, 0] = 1.0

    if nd > 0:                                                                      # for dominance model
        dominanceS = 999 * np.ones((nd,k), dtype=np.int)                            # indexes for dominant individuals
        dominanceS[0:ndseed, 0] = np.random.choice(seed, ndseed, replace=False)     # random sample nd dominant individuals
        probS[dominanceS[0:ndseed, 0], 0] = 1.0 + sigma

    probS[:, 0] = probS[:, 0] / ( seed + ndseed * sigma )                           # normalization

    ####################### Simulate until k villages are formed ####################

    while nv <= 50 and not done:
        if nv == 50:
            done = True
        ########################### Migration #######################
        if nv > 1:
            G1S = copy.copy(G0S)
            for v in range(k):
                if M[v] > 0:
                    pp = M[v]                                                  # number of ppl in village
                    mp = np.random.random((pp, 1))                             # migration probability
                    migrationS = [x for x in xrange(pp) if mp[x, 0] < m]       # index of ppl to be replaced by migrants
                    if len(migrationS) >= 1:
                        nei = [x for x in xrange(k) if x != v and M[x] > 0]    # index of neighbor villages with size>0
                        pool = G0S[:, nei]
                        pools0 = pool.ravel()
                        pools = [value for value in pools0 if value != 0]
                        G1S[migrationS, v] = np.random.choice(pools, len(migrationS), replace=False)
            G0S = copy.copy(G1S)                                               # activate migration

        G1S = np.zeros((N,k), dtype=np.int)

        ########################## Reproduction ######################
        for v in range(k):
            if M[v] > 0:
                pp1 = M[v]                                                      # population size for current generation
                pp2 = int( round( M[v] * (1 + r * ( 1 - M[v]/float(N) ) ) ) )   # population size of the next generation

            ###### choose father from previous generation ######
                if nd > 0:
                    DoS = []                                                                   # dominant son
                    while len(DoS) < int(round(pp2*delta)):                                    # redo if number of son of dominant less than pp2*delta
                        # record the number of offsprings each (of the N) father produced
                        numoffspringS = np.random.multinomial(pp2, probS[0:pp1, v], size=1)    # pp1 father --> pp2 son
                        zS = 0
                        DoS = []
                        for ii in range(pp1):
                            if numoffspringS[0, ii] > 0:
                                sonS = np.arange(zS, zS + numoffspringS[0, ii])
                                if ii in dominanceS[:,v]:                                      # if father is dominant
                                    DoS = np.concatenate((DoS, sonS))
                                G1S[sonS, v] = G0S[ii, v]
                                zS = zS + numoffspringS[0, ii]

                    nd_son = int(round(pp2*delta))
                    dominanceS[:nd_son,v] = np.random.choice(DoS, nd_son, replace=False)       # choose from DoS
                    dominanceS[nd_son:,v] = 999
                    nonDS = [x for x in xrange(pp2) if not x in dominanceS[:nd_son,v]]
                    rp = np.random.random((nd_son,1))                                          # replace probability
                    replaceS = [x for x in xrange(nd_son) if rp[x, 0] < p]                     # replace based on heritability
                    if len(replaceS) >= 1:
                        dominanceS[replaceS,v] = np.random.choice(nonDS, len(replaceS), replace=False)

                    probS[:pp2,v] = np.ones((pp2), dtype=float)
                    probS[pp2:,v] = 0.0
                    probS[dominanceS[0:nd_son, v], v] = 1.0 + sigma
                    probS[:,v] = probS[:,v] / float((pp2 + round(pp2*delta)*sigma))

                else:                                                        # non-dominance model
                    numoffspringS = np.random.multinomial(pp2, probS[0:pp1, v], size=1)
                    # produce pp2 number of offsprings, probS: prob for each (of pp1) father to be chosen, size: do the drawing 1 time
                    zS = 0                                                   # record index of son
                    for ii in range(pp1):                                    # go over all father
                        if numoffspringS[0,ii] > 0:                          # if father ii reproduce
                            sonS = np.arange(zS, zS + numoffspringS[0, ii])
                            G1S[sonS, v] = G0S[ii, v]                        # son haplotype follow father
                            zS = zS + numoffspringS[0, ii]

                    probS[:pp2,v] = np.ones(pp2, dtype=float) / float(pp2)
                    probS[pp2:,v] = 0.0

                M[v] = pp2

         ########################### Mutation #######################

        for v in range(k):
            if M[v] > 0:
                mup = np.random.random((M[v], 1))                             # mutation probability
                mutationS = [x for x in xrange(M[v]) if mup[x, 0] < mu]
                if len(mutationS) >= 1:
                    G1S[mutationS, v] = np.arange(1, len(mutationS)+1) + HS   # create new haplotypes
                    HS += len(mutationS)                                      # record total number of ever existed haplotype

        ########################### Founders #######################

        budding = np.random.permutation(k)                                    # randomize the budding order
        for v in budding:
            if M[v] >= Nbud:
                space = [x for x in range(k) if M[x]==0]                      # empty space
                if len(space) > 0:
                    new = space[0]
                    villagers = G1S[0:M[v],v]
                    founder = np.random.choice(M[v], seed, replace=False)     # random choose seed ppl
                    G1S[0:seed,new] = villagers[founder]                      # add founders to new village

                    index = [x for x in range(M[v]) if not x in founder]
                    G1S[0:M[v]-seed,v] = villagers[index]                     # remaining villagers
                    G1S[M[v]-seed:,v] = 0
                    if nd > 0:
                        Od = dominanceS[:,v]
                        Ld = [val for val in Od if val in index]              # remaining villagers who are dominance
                        ind2 = [dd for dd in range(len(index)) if index[dd] in Ld]

                        dominanceS[:,v] = 999
                        dominanceS[0:len(Ld),v] = ind2
                        probS[:,v] = 0.0
                        probS[0:M[v]-seed,v] = 1.0
                        probS[ind2,v] = 1.0 + sigma
                        probS[:,v] = probS[:,v] / np.sum(probS[:,v])

                        Nd_new = int(round(seed*delta))
                        dominanceS[0:Nd_new,new] = np.random.choice(seed, Nd_new, replace=False)
                        probS[0:seed,new] = 1.0
                        probS[dominanceS[0:Nd_new,new],new] = 1.0 + sigma
                        probS[:,new] = probS[:,new] / np.sum(probS[:,new])

                    else:
                        probS[:,v] = 0.0
                        probS[0:M[v]-seed,v] = 1.0 / float(M[v]-seed)

                        probS[:,new] = 0.0
                        probS[0:seed,new] = 1.0 / float(seed)

                    nv += 1
                    M[new] = seed
                    M[v] = M[v] - seed

        G0S = copy.copy(G1S)

    for v in range(k-5):
        villageS = G0S[0:M[v], v]
        sampS = np.random.choice(villageS, ns, replace=False)
        US, countsS = np.unique(sampS, return_counts=True)
        tyS = len(US)
        freqS = countsS / float(ns)
        sqS = np.power(freqS, 2)
        HetS[ens, v] = 1.0 - np.sum(sqS)

mhetS_v = np.mean(HetS, axis=1)              # mean heterozygosity over villages
mhetS = np.mean(mhetS_v, axis=0)             # mean heterozygosity over ensembles

print mhetS

#np.savetxt('FM_het_patri_mtDNA_mu0018.txt', HetS, fmt='%f')
np.savetxt('FM_het_matri_Y_mu0024.txt', HetS, fmt='%f')
