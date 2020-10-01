import numpy as np 
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.decomposition import TruncatedSVD


def random_flipping(graph_adj,epsilon):
	for i in range(graph_adj.shape[0]):
		for j in range(i+1,graph_adj.shape[1]):
			random_sample = np.random.uniform(0.0,1.0,1)
			if random_sample <= (1./ (1. + np.exp(epsilon))):
				graph_adj[i,j] = 1 - graph_adj[i,j]
				graph_adj[j,i] = graph_adj[i,j]

	return graph_adj

with open('citeseer/citeseer.node_labels','r') as f:
	node_label_segs = f.readlines(10000000000)

node_label = []
for k in range(len(node_label_segs)):
	segs = node_label_segs[k].split(',')
	node_label.append(int(segs[1]))

node_label = np.array(node_label)

citeseer_graph = np.zeros((node_label.shape[0],node_label.shape[0]))
with open('citeseer/citeseer.edges','r') as f:
	all_segs = f.readlines(100000000000)
	for k in range(len(all_segs)):
		segs = all_segs[k].split(',')
		s = int(segs[0])-1
		t = int(segs[1])-1
		citeseer_graph[s,t] = 1 
		citeseer_graph[t,s] = 1 

for k in range(citeseer_graph.shape[0]):
	citeseer_graph[k,k] = 1 

no_labels = len(np.unique(node_label))

delta = 1e-5 
epsilon = [1,10]

##### all the methods involved belong to input perturbation 
##### baseline 1: svd on original adjacency matrix 
clf_proj = TruncatedSVD(n_components = 3*no_labels,n_iter = 15,random_state=42)
graph_proj = clf_proj.fit_transform(citeseer_graph)
##### baseline 2: randomly flipping noise according to e^epsilon / (1 + e^epsilon)
d = 30
fraction = 8.
rand_proj = GaussianRandomProjection(n_components = d) 
graph_randn_proj = rand_proj.fit_transform(citeseer_graph)
noise_std = fraction * np.sqrt(1/d)
graph_randn_proj += np.random.normal(0.0,noise_std,size=graph_randn_proj.shape)


d = 30
fraction = 8.
delta = 1e-5
for alpha in range(2,100):
	#epsilon_renyi = np.max([2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction)))),2*(d/2*np.log((2+fraction)/(3+fraction)) + d/(2*(alpha-1))*np.log((2+fraction)/(alpha*(2+fraction) - (alpha-1)*(3+fraction))))])
	epsilon_renyi = 2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction))))

	#epsilon_renyi = 2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3)/(alpha*(3) - (alpha-1)*(2))))
	#epsilon_renyi = 2*(d/2*np.log(1.5) + d/(2*(alpha-1))*np.log((3)/(alpha*(3) - (alpha-1)*(2))))
	epsilon1 = epsilon_renyi + np.log(1/delta)/(alpha-1)
	print(epsilon1)

alpha = 3
epsilon_renyi = 2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction))))
epsilon1 = epsilon_renyi + np.log(1/delta)/(alpha-1)
print(epsilon1)

clf = RandomForestClassifier(n_estimators = 200)

#class_1 = np.where(node_label == 1)[0]
#class_2 = np.where(node_label == 2)[0]
#class_3 = np.where(node_label == 3)[0]
#class_4 = np.where(node_label == 4)[0]
#class_5 = np.where(node_label == 5)[0]
#class_6 = np.where(node_label == 6)[0]

#acc_score = []
#acc_score1 = []
#acc_score2 = []
acc_score_loo = []
loo_score = []
num_node = graph_proj.shape[0]
all_node_idx = np.zeros((num_node,))
for iround in range(num_node):
	all_node_idx = 0 * all_node_idx 
	all_node_idx[iround] = 1
	train_idx = np.where(all_node_idx < 1)[0]
	test_idx = np.where(all_node_idx > 0)[0]
	#clf.fit(graph_proj[train_idx,:],node_label[train_idx])
	#acc_score.append(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_1[train_idx,:],node_label[train_idx])
	#acc_score1.append(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_2[train_idx,:],node_label[train_idx])
	#acc_score2.append(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	clf.fit(graph_proj[train_idx,:],node_label[train_idx])
	acc_score_loo.append(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_randn_proj,node_label))

acc_score_loo_randn = []
loo_score_proj = []
num_node = graph_proj.shape[0]
all_node_idx = np.zeros((num_node,))
for iround in range(num_node):
	all_node_idx = 0 * all_node_idx 
	all_node_idx[iround] = 1
	train_idx = np.where(all_node_idx < 1)[0]
	test_idx = np.where(all_node_idx > 0)[0]
	#clf.fit(graph_proj[train_idx,:],node_label[train_idx])
	#acc_score.append(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_1[train_idx,:],node_label[train_idx])
	#acc_score1.append(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_2[train_idx,:],node_label[train_idx])
	#acc_score2.append(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	clf.fit(graph_randn_proj[train_idx,:],node_label[train_idx])
	acc_score_loo_randn.append(clf.score(graph_randn_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_randn_proj,node_label))

##### count out/in degree of each node 
degree_per_node = []
for k in range(citeseer_graph.shape[0]):
	degree_per_node.append(np.sum(citeseer_graph[k,:]))

mis_classified_idx = np.where(np.array(acc_score_loo) < 1)[0]
print(len(np.where(np.array(degree_per_node) >= 4)[0]))
print(len(np.where(np.array(degree_per_node)[mis_classified_idx] >= 4)[0]))
print(len(np.where(np.array(degree_per_node) < 4)[0]))
print(len(np.where(np.array(degree_per_node)[mis_classified_idx] < 4)[0]))

##### see if acc varies proportionally with respect to the node-wise degree
import matplotlib.pyplot as pyplot
pyplot.plot(acc_score_loo,degree_per_node)

'''
In [31]: clean_score_high_degree
Out[31]: 0.7443820224719101

In [32]: clean_score_low_degree
Out[32]: 0.5

In [33]: rand_score_low_degree
Out[33]: 0.4376959247648903

In [34]: rand_score_high_degree
Out[34]: 0.49297752808988765

In [35]: np.mean(acc_score_loo_randn)
Out[35]: 0.4497549019607843

In [36]: np.mean(acc_score_loo)
Out[36]: 0.5533088235294118
node degree <=4 and node degree >4 
In [37]: 
'''