import numpy as np 
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.decomposition import TruncatedSVD


def random_flipping(graph_adj,epsilon):
	for i in range(graph_adj.shape[0]):
		for j in range(i+1,graph_adj.shape[1]):
			random_sample = np.random.uniform(0.0,1.0,1)
			if random_sample <= (1-1 / (1 + np.exp(epsilon))):
				graph_adj[i,j] = 1 - graph_adj[i,j]
				graph_adj[j,i] = graph_adj[i,j]

	return graph_adj

with open('cora/cora.node_labels','r') as f:
	node_label_segs = f.readlines(10000000000)

node_label = []
for k in range(len(node_label_segs)):
	segs = node_label_segs[k].split(',')
	node_label.append(int(segs[1]))

node_label = np.array(node_label)

citeseer_graph = np.zeros((node_label.shape[0],node_label.shape[0]))
with open('cora/cora.edges','r') as f:
	all_segs = f.readlines(100000000000)
	for k in range(len(all_segs)):
		segs = all_segs[k].split(',')
		s = int(segs[0])-1
		t = int(segs[1])-1
		citeseer_graph[s,t] = 1 
		citeseer_graph[t,s] = 1 

no_labels = len(np.unique(node_label))

delta = 1e-5 
epsilon = [1,10]

##### all the methods involved belong to input perturbation 
##### baseline 1: svd on original adjacency matrix 
clf_proj = TruncatedSVD(n_components = 3*no_labels,n_iter = 15,random_state=42)
graph_proj = clf_proj.fit_transform(citeseer_graph)
##### baseline 2: randomly flipping noise according to e^epsilon / (1 + e^epsilon)
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

##### count out/in degree of each node 
degree_per_node = []
for k in range(citeseer_graph.shape[0]):
	degree_per_node.append(np.sum(citeseer_graph[k,:]))

mis_classified_idx = np.where(np.array(acc_score_loo) < 1)[0]
print(np.quantile(np.array(degree_per_node)[mis_classified_idx],[0.1,0.3,0.5,0.7,0.95]))
print(len(np.where(np.array(degree_per_node) >= 4)[0]))
print(len(np.where(np.array(degree_per_node)[mis_classified_idx] >= 4)[0]))
print(len(np.where(np.array(degree_per_node) < 4)[0]))
print(len(np.where(np.array(degree_per_node)[mis_classified_idx] < 4)[0]))

for k in range(30,35):
    print(np.mean(np.array(acc_score_loo)[np.where(np.array(degree_per_node) >= k)[0]])) #/ float(len(np.where(np.array(degree_per_node) >= k)[0])))
##### see if acc varies proportionally with respect to the node-wise degree
import matplotlib.pyplot as pyplot
pyplot.plot(acc_score_loo,degree_per_node)

'''
In [74]: 0.50/0.69
Out[74]: 0.7246376811594204

In [75]: 0.53/0.81
Out[75]: 0.654320987654321

In [76]: len(low_node_degree_idx)
Out[76]: 1068

In [77]: len(high_node_degree_idx)
Out[77]: 1640

low_node_degree_idx = np.where(np.array(degree_per_node) <= 3)[0]
high_node_degree_idx = np.where(np.array(degree_per_node) > 3)[0]
In [78]: clean_score_low_degree
Out[78]: 0.6891385767790262

In [79]: clean_score_high_degree
Out[79]: 0.8060975609756098

In [80]: rand_score_low_degree
Out[80]: 0.49719101123595505

In [81]: rand_score_high_degree
Out[81]: 0.5378048780487805
'''