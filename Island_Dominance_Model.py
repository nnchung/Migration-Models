import matplotlib.pyplot as plt
import numpy as np
import copy

############################## Parameters #############################

N = 300                                               # population size
k = 50                                                # number of village
mu = 0.0249                                           # mutation rate
m = 0.005                                             # migration rate
p = 0.0                                               # 1 - heritability
sigma = 1.0                                           # reproductive advantage
delta = 0.06                                          # fraction of dominance (set to zero for nondominance island model)
nd = int(round(N*delta))                              # number of dominant individuals

T = 100                                               # number of generation
ns = 50                                               # sampling size
Nens = 5                                              # number of realizations

############################# Initialization ###########################

populationS = np.zeros((N, k), dtype=np.int)          # single haplotype
for j in range(k):
    populationS[:, j] = [j+1 for i in xrange(N)]

populationD = np.zeros((N, k), dtype=np.int)          # diverse haplotype
for j in range(k):
    populationD[:, j] = [j*N+i+1 for i in xrange(N)]

HapCountS = np.zeros((T, Nens, k), dtype=np.int)      # count of haplotype
HapCountD = np.zeros((T, Nens, k), dtype=np.int)
HetS = np.zeros((T, Nens, k), dtype=np.float)         # heterozygosity of haplotype
HetD = np.zeros((T, Nens, k), dtype=np.float)

############################### Simulation ###############################

for ens in range(Nens):
    print ens
    t = 0

    ########################## Population S #############################

    G0S = copy.copy(populationS)                      # current generation
    G1S = copy.copy(populationS)                      # updated generation
    HS = k                                            # total number of haplotype in all villages
    probS = np.ones((N,k), dtype=float)               # probability to reproduce

    if nd > 0:                                        # for dominance model
        dominanceS = np.zeros((nd,k), dtype=np.int)   # indexes for dominant individuals
        for v in range(k):                            # random sample nd dominant individuals
            dominanceS[:, v] = np.random.choice(N, nd, replace=False)
            probS[dominanceS[:, v], v] = 1.0 + sigma

    probS = probS / ( N * (1 + delta * sigma) )       # normalization

    ########################## Population D ##############################

    G0D = copy.copy(populationD)
    G1D = copy.copy(populationD)
    HD = N*k                                          # there are Nk haplotypes in all villages
    probD = np.ones((N,k), dtype=float)

    if nd > 0:
        dominanceD = np.zeros((nd,k), dtype=np.int)
        for v in range(k):
            dominanceD[:,v] = np.random.choice(N, nd, replace=False)
            probD[dominanceD[:, v], v] = 1.0 + sigma  # reproductive prob for dominant: 1+sigma

    probD = probD / ( N * (1 + delta * sigma) )

    ####################### Simulate for T generations ####################

    while t < T:

        for v in range(k):
            villageS = G0S[:, v]
            sampS = np.random.choice(villageS, ns, replace=False)
            US, countsS = np.unique(sampS, return_counts=True)
            tyS = len(US)
            HapCountS[t, ens, v] = tyS
            freqS = countsS / float(ns)
            sqS = np.power(freqS, 2)
            HetS[t, ens, v] = 1.0 - np.sum(sqS)

            ########### calculate number of haplotype & heterozygosity ######

            villageD = G0D[:, v]
            sampD = np.random.choice(villageD, ns, replace=False)
            UD, countsD = np.unique(sampD, return_counts=True)
            tyD = len(UD)
            HapCountD[t, ens, v] = tyD
            freqD = countsD / float(ns)
            sqD = np.power(freqD, 2)
            HetD[t, ens, v] = 1.0 - np.sum(sqD)

        for v in range(k):
            ###### choose father from previous generation ######
            if nd > 0:
                DoS = []                        # dominant son
                while len(DoS) < nd:            # redo if number of son of dominant less than nd
                    # record the number of offsprings each (of the N) father produced
                    numoffspringS = np.random.multinomial(N, probS[:, v], size=1)
                    zS = 0
                    DoS = []
                    for ii in range(N):
                        if numoffspringS[0, ii] > 0:
                            sonS = np.arange(zS, zS + numoffspringS[0, ii])
                            if ii in dominanceS[:,v]:                 # if father is dominant
                                DoS = np.concatenate((DoS, sonS))
                            G1S[sonS, v] = G0S[ii, v]
                            zS = zS + numoffspringS[0, ii]

                dominanceS[:,v] = np.random.choice(DoS, nd, replace=False)   # choose from DoS
                nonDS = [x for x in xrange(N) if not x in dominanceS[:,v]]
                rp = np.random.random((nd,1))                                # replace probability
                replaceS = [x for x in xrange(nd) if rp[x, 0] < p]   # replace based on heritability
                if len(replaceS) >= 1:
                    dominanceS[replaceS,v] = np.random.choice(nonDS, len(replaceS), replace=False)

                probS[:,v] = np.ones((N))
                probS[dominanceS[:, v], v] = 1.0 + sigma

            else:                                                     # non-dominance model
                numoffspringS = np.random.multinomial(N, probS[:, v], size=1)
                # produce N number of offsprings, probS: prob for each (of N) father to be chosen, size: do the drawing 1 time
                zS = 0                                                # record index of son
                for ii in range(N):                                   # go over all father
                    if numoffspringS[0,ii] > 0:                       # if father ii reproduce
                        sonS = np.arange(zS, zS + numoffspringS[0, ii])
                        G1S[sonS, v] = G0S[ii, v]                     # son haplotype follow father
                        zS = zS + numoffspringS[0, ii]

            if nd > 0:
                DoD = []
                while len(DoD) < nd:
                    numoffspringD = np.random.multinomial(N, probD[:,v], size=1)
                    zD = 0
                    DoD = []
                    for ii in range(N):
                        if numoffspringD[0,ii] > 0:
                            sonD = np.arange(zD, zD+numoffspringD[0,ii])
                            if ii in dominanceD[:,v]:
                                DoD = np.concatenate((DoD, sonD))
                            G1D[sonD,v] = G0D[ii,v]
                            zD = zD+numoffspringD[0,ii]

                dominanceD[:, v] = np.random.choice(DoD, nd, replace=False)
                nonDD = [x for x in xrange(N) if not x in dominanceD[:,v]]
                rp = np.random.random((nd, 1))
                replaceD = [x for x in xrange(nd) if rp[x, 0] < p]
                if len(replaceD) >= 1:
                    dominanceD[replaceD,v] = np.random.choice(nonDD, len(replaceD), replace=False)

                probD[:, v] = np.ones((N))
                probD[dominanceD[:, v], v] = 1.0 + sigma

            else:
                numoffspringD = np.random.multinomial(N, probD[:, v], size=1)
                zD = 0
                for ii in range(N):
                    if numoffspringD[0,ii] > 0:
                        sonD = np.arange(zD, zD + numoffspringD[0, ii])
                        G1D[sonD,v] = G0D[ii,v]
                        zD = zD+numoffspringD[0,ii]

        if nd > 0:                                    # normalization of probability for D model
            probS = probS / (N*(1+delta*sigma))
            probD = probD / (N*(1+delta*sigma))

        ########################### Migration #######################

        for v in range(k):
            mp = np.random.random((N, 1))                             # migration probability
            migrationS = [x for x in xrange(N) if mp[x, 0] < m]       # index of ppl to be replaced by migrants
            if len(migrationS) >= 1:
                nei = [x for x in xrange(k) if x != v]                # neighboring villages
                pool = G0S[:, nei]
                pools = pool.ravel()
                G1S[migrationS, v] = np.random.choice(pools, len(migrationS), replace=False)

            mp = np.random.random((N,1))
            migrationD = [x for x in xrange(N) if mp[x, 0] < m]
            if len(migrationD) >= 1:
                nei = [x for x in xrange(k) if x != v]
                pool = G0D[:, nei]
                pools = pool.ravel()
                G1D[migrationD, v] = np.random.choice(pools, len(migrationD), replace=False)

         ########################### Mutation #######################

        for v in range(k):
            mup = np.random.random((N, 1))                               # mutation probability
            mutationS = [x for x in xrange(N) if mup[x, 0] < mu]
            if len(mutationS) >= 1:
                G1S[mutationS, v] = np.arange(1, len(mutationS)+1) + HS  # create new haplotypes
                HS += len(mutationS)                                     # record total number of ever existed haplotype

            mup = np.random.random((N,1))
            mutationD = [x for x in xrange(N) if mup[x, 0] < mu]
            if len(mutationD) >= 1:
                G1D[mutationD, v] = np.arange(1, len(mutationD)+1) + HD
                HD += len(mutationD)

        G0S = copy.copy(G1S)
        G0D = copy.copy(G1D)
        t += 1


mhcS_v = np.mean(HapCountS, axis=2)          # mean haplo counts over villages
mhcD_v = np.mean(HapCountD, axis=2)
mhetS_v = np.mean(HetS, axis=2)              # mean heterozygosity over villages
mhetD_v = np.mean(HetD, axis=2)

mhcS = np.mean(mhcS_v, axis=1)               # mean haplo counts over ensembles
mhcD = np.mean(mhcD_v, axis=1)
mhetS = np.mean(mhetS_v, axis=1)             # mean heterozygosity over ensembles
mhetD = np.mean(mhetD_v, axis=1)

fig = plt.figure(num=1, figsize=(10, 4.5), dpi=100, facecolor='w', edgecolor='k')

ax1 = fig.add_subplot(121)
plt.plot(np.arange(1,T+1), mhcS)
plt.plot(np.arange(1,T+1), mhcD)
plt.xlabel('Generation')
plt.ylabel('Haplotype count')
plt.text(40, 1.1*ns, r'Parameters: $N=%.0f$, $\mu=%.4f$, $m=%.3f$, $k=%.0f$, $p=%.1f$, $\sigma=%.1f$, $\delta=%.2f$' %(N, mu, m, k, p, sigma, delta), fontsize=10)

ax2 = fig.add_subplot(122)
plt.plot(np.arange(1,T+1), mhetS)
plt.plot(np.arange(1,T+1), mhetD)
plt.xlabel('Generation')
plt.ylabel('Heterozygosity')

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.92, hspace=0.25, wspace=0.25)
plt.show()



