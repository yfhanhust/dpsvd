import numpy as np 
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import qr 
from sklearn.cluster import KMeans

def random_flipping(graph_adj,epsilon):
	for i in range(graph_adj.shape[0]):
		for j in range(i+1,graph_adj.shape[1]):
			random_sample = np.random.uniform(0.0,1.0,1)
			if random_sample <= (1./ (1. + np.exp(epsilon))):
				graph_adj[i,j] = 1 - graph_adj[i,j]
				graph_adj[j,i] = graph_adj[i,j]

	return graph_adj

def local_degree_gen(graph_adj,node_id,local_group_idx):
	no_local_group = len(local_group_idx)
	local_degree_vec = []
	for k in range(no_local_group):
		local_degree_vec.append(np.sum(graph_adj[node_id,local_group_idx[k]]))

	return local_degree_vec

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

no_labels = len(np.unique(node_label))

###### Step.1 initialize with randomly grouped nodes 
n_cluster = 200
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

######## Step.2 local node degree calculated using the randomly grouped nodes
local_node_degree = []
for k in range(len(node_label)):
	degree_no = local_degree_gen(citeseer_graph,k,group_idx_range)
	degree_no += np.random.laplace(loc=0.0,scale=2.,size=len(degree_no)) 
	local_node_degree.append(np.array(degree_no))

local_node_degree = np.array(local_node_degree)

##### Step.3 regroup the nodes with the local node degree vectors and specified node group numbers 
n_cluster = 500
clustering_ml = KMeans(n_clusters=n_cluster).fit(local_node_degree)
cluster_idx = clustering_ml.labels_

group_idx_range = []
for k in range(n_cluster):
	group_idx_range.append(np.where(cluster_idx == k)[0])

######## Step.4 calcualte local node degree
local_node_degree = []
for k in range(len(node_label)):
	degree_no = local_degree_gen(citeseer_graph,k,group_idx_range)
	degree_no += np.random.laplace(loc=0.0,scale=2.,size=len(degree_no))
	local_node_degree.append(np.array(degree_no))

local_node_degree = np.array(local_node_degree)

citeseer_graph_copy = np.copy(citeseer_graph)
citeseer_graph_copy = random_flipping(citeseer_graph_copy,1.0)
graph_proj_1 = clf_proj.fit_transform(citeseer_graph_copy)



'''
delta = 1e-5 
epsilon = [1,10]

##### all the methods involved belong to input perturbation 
##### baseline 1: svd on original adjacency matrix 
clf_proj = TruncatedSVD(n_components = 3*no_labels,n_iter = 15,random_state=42)
graph_proj = clf_proj.fit_transform(citeseer_graph)
##### baseline 2: randomly flipping noise according to e^epsilon / (1 + e^epsilon)
citeseer_graph_copy = np.copy(citeseer_graph)
citeseer_graph_copy = random_flipping(citeseer_graph_copy,1.0)
graph_proj_1 = clf_proj.fit_transform(citeseer_graph_copy)

##### baseline 3: add Gaussian noise to the graph adjacency matrix directly 
citeseer_graph_copy2 = np.copy(citeseer_graph)
noise_std = 2.4
alpha = 20.
epsilon1 = alpha / (noise_std * noise_std) + np.log(1/delta)/(alpha-1)
print(epsilon1)
gaussian_graph_noise = np.random.normal(0,noise_std,citeseer_graph_copy2.shape)
citeseer_graph_copy2 += gaussian_graph_noise 
graph_proj_2 = clf_proj.fit_transform(citeseer_graph_copy2)




##### baseline 4: gradient descent + gradient perturbation 


##### the proposed method: random projection + add Gaussian noise to the graph adjacency matrix directly 
#d = 10
d = 30
fraction = 8.
rand_proj = GaussianRandomProjection(n_components = d) 
graph_randn_proj = rand_proj.fit_transform(citeseer_graph)
noise_std = fraction * np.sqrt(1/d)
graph_randn_proj += np.random.normal(0.0,noise_std,size=graph_randn_proj.shape)
quad_base, r = qr(graph_randn_proj)
#graph_randn_svd = quad_base[:,:3*no_labels]
graph_randn_svd = graph_randn_proj
#alpha = 2.5
#epsilon_renyi = np.max([2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction)))),2*(d/2*np.log((2+fraction)/(3+fraction)) + d/(2*(alpha-1))*np.log((2+fraction)/(alpha*(2+fraction) - (alpha-1)*(3+fraction))))])
#epsilon1 = epsilon_renyi + np.log(1/delta)/(alpha-1)
#print(epsilon1)
'''
'''
d = 10
fraction = 5.
for alpha in [2,3,4,5,10,15,20,25,30,35,40,100]:
	#epsilon_renyi = np.max([2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction)))),2*(d/2*np.log((2+fraction)/(3+fraction)) + d/(2*(alpha-1))*np.log((2+fraction)/(alpha*(2+fraction) - (alpha-1)*(3+fraction))))])
	epsilon_renyi = 2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction))))

	#epsilon_renyi = 2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3)/(alpha*(3) - (alpha-1)*(2))))
	#epsilon_renyi = 2*(d/2*np.log(1.5) + d/(2*(alpha-1))*np.log((3)/(alpha*(3) - (alpha-1)*(2))))
	epsilon1 = epsilon_renyi + np.log(1/delta)/(alpha-1)
	print(epsilon1)
'''


clf = RandomForestClassifier(n_estimators = 200)

class_1 = np.where(node_label == 1)[0]
class_2 = np.where(node_label == 2)[0]
class_3 = np.where(node_label == 3)[0]
class_4 = np.where(node_label == 4)[0]
class_5 = np.where(node_label == 5)[0]
class_6 = np.where(node_label == 6)[0]

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
	clf.fit(graph_proj_1[train_idx,:],node_label[train_idx])
	acc_score.append(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_1[train_idx,:],node_label[train_idx])
	#acc_score1.append(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_2[train_idx,:],node_label[train_idx])
	#acc_score2.append(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	clf.fit(local_node_degree[train_idx,:],node_label[train_idx])
	acc_score_randn.append(clf.score(local_node_degree[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_randn_proj,node_label))