import numpy as np 
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.decomposition import TruncatedSVD


def random_flipping(graph_adj,epsilon):
	for i in range(graph_adj.shape[0]):
		for j in range(i+1,graph_adj.shape[1]):
			random_sample = np.random.uniform(0.0,1.0,1)
			if random_sample <= 1 / (1 + np.exp(epsilon)):
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

##### all the methods involved belong to input perturbation 
##### baseline 1: svd on original adjacency matrix 
clf_proj = TruncatedSVD(n_components = 3*no_labels,n_iter = 15,random_state=42)
graph_proj = clf_proj.fit_transform(citeseer_graph)

###### Step.1 random grouping 
n_cluster = 35
node_idx = np.array(range(len(node_label)))
np.random.shuffle(node_idx)
group_size = int(float(len(node_label)) / float(n_cluster))
group_idx_range = []
for i in range(n_cluster):
	start_idx = i*group_size
	end_idx = i*group_size + group_size
	if end_idx > (len(node_label) - 1):
		end_idx = len(node_label)-1 

	group_idx_range.append(np.array(node_idx)[start_idx:end_idx])



######## clean baseline 
#clf_proj = TruncatedSVD(n_components = 30,n_iter = 15,random_state=42)
#graph_proj = clf_proj.fit_transform(citeseer_graph)

######## Step.2 calculate local node degree vectors with the randomly initliazed node groups
local_node_degree = []
for k in range(len(node_label)):
	degree_no = local_degree_gen(citeseer_graph,k,group_idx_range)
	degree_no += np.random.laplace(loc=0.0,scale=1.,size=len(degree_no)) 
	local_node_degree.append(np.array(degree_no))

local_node_degree = np.array(local_node_degree)


####### Step.3 Refine the node clusters with the local node degree vectors derived from Step.2 
n_cluster = 10
clustering_ml = KMeans(n_clusters=n_cluster).fit(local_node_degree)
cluster_idx = clustering_ml.labels_

group_idx_range = []
for k in range(n_cluster):
	group_idx_range.append(np.where(cluster_idx == k)[0])

######## Step.4 Calculate local node degree vectors with the updated node groups 
local_node_degree = []
for k in range(len(node_label)):
	degree_no = local_degree_gen(citeseer_graph,k,group_idx_range)
	degree_no += np.random.laplace(loc=0.0,scale=1.,size=len(degree_no))
	local_node_degree.append(np.array(degree_no))

local_node_degree = np.array(local_node_degree)

clf = RandomForestClassifier(n_estimators = 200)

class_1 = np.where(node_label == 1)[0]
class_2 = np.where(node_label == 2)[0]
class_3 = np.where(node_label == 3)[0]
class_4 = np.where(node_label == 4)[0]
class_5 = np.where(node_label == 5)[0]
class_6 = np.where(node_label == 6)[0]
class_7 = np.where(node_label == 7)[0]


acc_score = []
acc_score1 = []
acc_score2 = []
acc_score_randn = [] 
for iround in range(10):
	train_idx = []
	test_idx = []
	np.random.shuffle(class_1)
	np.random.shuffle(class_2)
	np.random.shuffle(class_3)
	np.random.shuffle(class_4)
	np.random.shuffle(class_5)
	np.random.shuffle(class_6)
	train_idx.extend(class_1[:int(0.9*len(class_1))])
	train_idx.extend(class_2[:int(0.9*len(class_2))])
	train_idx.extend(class_3[:int(0.9*len(class_3))])
	train_idx.extend(class_4[:int(0.9*len(class_4))])
	train_idx.extend(class_5[:int(0.9*len(class_5))])
	train_idx.extend(class_6[:int(0.9*len(class_6))])
	test_idx.extend(class_1[int(0.9*len(class_1)):])
	test_idx.extend(class_2[int(0.9*len(class_2)):])
	test_idx.extend(class_3[int(0.9*len(class_3)):])
	test_idx.extend(class_4[int(0.9*len(class_4)):])
	test_idx.extend(class_5[int(0.9*len(class_5)):])
	test_idx.extend(class_6[int(0.9*len(class_6)):])
	clf.fit(graph_proj[train_idx,:],node_label[train_idx])
	acc_score.append(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	clf.fit(local_node_degree[train_idx,:],node_label[train_idx])
	acc_score_randn.append(clf.score(local_node_degree[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_randn_proj,node_label))