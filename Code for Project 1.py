'''
Code from Markus Ferdinand Dablander
'''

import numpy as np
import math 
import networkx as nx


def network(ts):
	'''takes list containing a time series and produces the adjacency matrix of the resulting horizontal visibility network
	'''
	n = len(ts)
	A = np.zeros((n,n),int)
	for i in range(0,n-1):
		for j in range(i+1,n):
			if j==i+1:
				A[i,j]=1
			elif j>i+1:
				if np.amax(ts[i+1:j])<np.amin([ts[i],ts[j]]):
					A[i,j]=1	
			if ts[j]>=ts[i]:
				break
	return A+np.transpose(A)


def overlaplist(matrixlist):
	'''takes a list of adjacency matrices of horizontal visibility networks and computes the average edge overlap of the corresponding multilayer network
	'''
	m = len(matrixlist)
	n = len(matrixlist[0])
	num = 0
	denom = 0
	for i in range(n):
		for j in range(i+1,n):
			alphasum = 0
			for alpha in range(m):
				alphasum += matrixlist[alpha][i,j]
			num += alphasum
			if alphasum == 0:
				delta = 1
			else:
				delta = 0
			denom += 1-delta
	denom *= m
	return num/denom


def overlap(*matrixlist):
	return overlaplist(matrixlist)


def degentropy(A1): 
''' takes the adjacency matrix of a network and computes the entropy of the degree distribution
'''
	entropy=0
	n = np.shape(A1)[1]
	maxdeg1 = np.amax(np.sum(A1, axis=1))
	for k1 in range(maxdeg1+1):
		nk1=0
		for i in np.sum(A1,axis=1):
			if i == k1:
				nk1+=1
		if nk1!=0:
			Pk1=nk1/n
			entropy += -Pk1*math.log(Pk1)
	return entropy


def degentropyavlist(matrixlist):
	degentropyav = 0
	m = len(matrixlist)
	for k in range(m):
		degentropyav += degentropy(matrixlist[k])
	return degentropyav/m

def degentropyav(*matrixlist):
	return degentropyavlist(matrixlist)	


def mutinf(A1, A2):	
''' takes the adjacency matrices of two networks and computes the mutual information of the degree distributions of those two networks
'''
	n = np.shape(A1)[1]
	I=0
	maxdeg1=np.amax(np.sum(A1, axis=1))
	maxdeg2=np.amax(np.sum(A2, axis=1))
	for k1 in range(maxdeg1+1):
		for k2 in range(maxdeg2+1):
			nk1=0
			nk2=0
			nk1k2=0
			rowsumA1 = np.sum(A1,axis=1)
			rowsumA2 = np.sum(A2,axis=1)
			for i in range(n):
				if rowsumA1[i] == k1:
					nk1+=1
				if rowsumA2[i] == k2:
					nk2+=1
				if rowsumA1[i] == k1 and rowsumA2[i]==k2:
					nk1k2+=1
			if nk1!=0 and nk2!=0 and nk1k2!=0:
				Pk1=nk1/n
				Pk2=nk2/n
				Pk1k2=nk1k2/n
				I += Pk1k2*math.log(Pk1k2/(Pk1*Pk2))
	return I


def mutinfavlist(matrixlist):
	mutinfav = 0
	m = len(matrixlist)
	for alpha in range(m):
		for beta in range(alpha+1,m):
			mutinfav += mutinf(matrixlist[alpha],matrixlist[beta])
	return mutinfav/(m*(m-1)/2)


def mutinfav(*matrixlist):
	return mutinvavlist(matrixlist)	


def networkanalysis(*listts):
''' takes a list of network adjacency matrices that together form a multilayer network and combines all the above functions and spits out a list with results
'''
	m = len(listts)	
	networkmatrices=list(range(m))
	for k in range(m):
		networkmatrices[k] = network(listts[k])
	overlaparray = np.zeros((m,m),float)
	for alpha in range(m):
		for beta in range(alpha+1,m):
			overlaparray[alpha,beta] = overlap(networkmatrices[alpha], networkmatrices[beta])
	overlaparray += np.transpose(overlaparray)
	o = overlaplist(networkmatrices)
	entropyarray = np.zeros((1,m),float)
	degentropyav=0
	for alpha in range(m):
		entropyarray[0,alpha] = degentropy(networkmatrices[alpha])
		degentropyav += entropyarray[0,alpha]
	degentropyav = degentropyav/m
	informationarray = np.zeros((m,m), float)
	mutinfav = 0
	for alpha in range(m):
		for beta in range(alpha+1,m):
			informationarray[alpha,beta] = mutinf(networkmatrices[alpha], networkmatrices[beta])
			mutinfav += informationarray[alpha,beta]
	informationarray += np.transpose(informationarray)	
	mutinfav = mutinfav/(m*(m-1)/2)
	norminformationarray = np.zeros((m,m), float)
	for alpha in range(m):
		for beta in range(alpha+1,m):
			norminformationarray[alpha,beta]  = informationarray[alpha,beta] / (entropyarray[0,alpha]+entropyarray[0,beta]-informationarray[alpha,beta])
	mutinfavnorm = np.average(norminformationarray)
	norminformationarray += np.transpose(norminformationarray)		
	#print("OUTPUT:\n\n 0 ListofNetworkmatrices,\n 1 Overlaparray,\n 2 AverageOverlap,\n 3 Entropyarray,\n 4 AverageEntropy,\n 5 MutualInformationArray,\n 6 AverageMutualInformation, \n 7 MutualNormalizedInformationArray, \n 8 AverageMutualInformationNormalized\n \n")
	listresults = [networkmatrices,overlaparray,o,entropyarray,degentropyav,informationarray,mutinfav, norminformationarray, mutinfavnorm]
	return listresults

	'''
	OUTPUT OF networkanalysis()
	0 ListofNetworkmatrices
	1 Overlaparray
	2 AverageOverlap
	3 EntropyArray
	4 AverageEntropy
	5 MutualInformationArray
	6 AverageMutualInformation
	7 MutualNormalizedInformationArray
	8 AverageMutualInformationNormalized
	9 Lac/Naa ratio
	10 Filename

	'''


	def powerlist(n):
    pl = []
    for i in range(1,1 << n):
        pl.append([j for j in range(n) if (i & (1 << j))])
    return pl




def network_features(s1,s2,s3):
	'''Takes three signals as input (s1, s2 and s3). Each signal has the form of a list or a 1-dimensional numpy-array.
	Each signal is transformed into a horizontal visibility network G (which takes the form of a 2-dimensional numpy adjacency matrix with entries 0 and 1).
	Then the average edge overlap AEO and the mutual information of degree distribution MID is computed for all possible pairs of networks G.
	The function returns a feature vector of the form [AEO(G1,G2),AEO(G1,G3),AEO(G2,G3),MID(G1,G2),MID(G1,G3),MID(G2,G3)].
	'''

	G1 = network(s1)
	G2 = network(s2)
	G3 = network(s3)

	features=np.zeros((1,6))[0]

	features[0] = overlap(G1,G2)
	features[1] = overlap(G1,G3)
	features[2] = overlap(G2,G3)
	features[3] = mutinf(G1,G2)
	features[4] = mutinf(G1,G3)
	features[5] = mutinf(G2,G3)

	return features

