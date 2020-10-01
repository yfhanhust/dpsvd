import numpy as np 
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import qr 
from sklearn.utils import check_random_state
from sklearn.extmath import randomized_range_finder 

def random_flipping(graph_adj,epsilon):
	for i in range(graph_adj.shape[0]):
		for j in range(i+1,graph_adj.shape[1]):
			random_sample = np.random.uniform(0.0,1.0,1)
			if random_sample <= 1 / (1 + np.exp(epsilon)):
				graph_adj[i,j] = 1 - graph_adj[i,j]
				graph_adj[j,i] = graph_adj[i,j]

	return graph_adj

with open('citeseer/citeseer.node_labels','r') as f:
	node_label_segs = f.readlines(10000000000)

def singlepass_evd(graph_adj,k):
	l = k + 100
	n = graph_adj.shape[0]
	random_state = check_random_state(42)
	#Omg = np.random.randn(n,l)
	Omega = random_state.normal(size=(n,l))
	Qhat,Rhat = np.linalg.qr(np.dot(graph_adj,Omega))
	Qhat_pr,Rhat_pr = np.linalg.qr(np.dot(graph_adj.T,Omega))
	#Qhat = randomized_range_finder(graph_adj,size=l,n_iter=10,power_iteration_normalizer='QR')
	Y = np.dot(graph_adj,Omega)
	Yt = np.dot(graph_adj.T,Omega)
	Q = Qhat[:,0:k]
	Qt = Qhat_pr[:,0:k]
	Coef = Rhat_pr[0:k,:].T
	Target = np.dot(Omega.T,Q)
	print(Coef.shape)
	print(Target.shape)
	#Qhat, R = np.linalg.qr(Y)
	#B = np.dot(Qhat.T,graph_adj)
	Uhat = np.linalg.lstsq(Coef,Target)
	print(Uhat[0].shape)
	u,s,v = np.linalg.svd(Uhat[0])
	graph_randn_svd = np.dot(Q,u)
	return graph_randn_svd[:,:k]

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

delta = 1e-5 
epsilon = [1,10]

##### all the methods involved belong to input perturbation 
##### baseline 1: svd on original adjacency matrix 
clf_proj = TruncatedSVD(n_components = 30,n_iter = 15,random_state=42)
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
'''
rand_proj = GaussianRandomProjection(n_components = d) 
graph_randn_proj = rand_proj.fit_transform(citeseer_graph)
noise_std = fraction * np.sqrt(1/d)
graph_randn_proj += np.random.normal(0.0,noise_std,size=graph_randn_proj.shape)
quad_base, r = qr(graph_randn_proj)
#graph_randn_svd = quad_base[:,:3*no_labels]
graph_randn_svd = graph_randn_proj
'''
graph_randn_svd = singlepass_evd(citeseer_graph,d)
#alpha = 2.5
#epsilon_renyi = np.max([2*(d/2*np.log((3+fraction)/(2+fraction)) + d/(2*(alpha-1))*np.log((3+fraction)/(alpha*(3+fraction) - (alpha-1)*(2+fraction)))),2*(d/2*np.log((2+fraction)/(3+fraction)) + d/(2*(alpha-1))*np.log((2+fraction)/(alpha*(2+fraction) - (alpha-1)*(3+fraction))))])
#epsilon1 = epsilon_renyi + np.log(1/delta)/(alpha-1)
#print(epsilon1)
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
from sklearn.utils.extmath import randomized_svd,randomized_range_finder 
u,s,v = randomized_svd(citeseer_graph,n_components=30)
graph_randn_svd = u

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
	#clf.fit(graph_proj[train_idx,:],node_label[train_idx])
	#acc_score.append(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_1[train_idx,:],node_label[train_idx])
	#acc_score1.append(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_1[test_idx,:],node_label[test_idx]))
	#clf.fit(graph_proj_2[train_idx,:],node_label[train_idx])
	#acc_score2.append(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_proj_2[test_idx,:],node_label[test_idx]))
	clf.fit(graph_randn_svd[train_idx,:],node_label[train_idx])
	acc_score_randn.append(clf.score(graph_randn_svd[test_idx,:],node_label[test_idx]))
	#print(clf.score(graph_randn_proj,node_label))

'''
###### clustering 
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score,normalized_mutual_info_score,v_measure_score,mutual_info_score

cluster_model = KMeans(n_clusters=no_labels).fit(graph_proj)
print(normalized_mutual_info_score(node_label,cluster_model.labels_))

cluster_model = KMeans(n_clusters=no_labels).fit(graph_proj_1)
print(normalized_mutual_info_score(node_label,cluster_model.labels_))

cluster_model = KMeans(n_clusters=no_labels).fit(graph_proj_2)
print(normalized_mutual_info_score(node_label,cluster_model.labels_))

cluster_model = KMeans(n_clusters=no_labels).fit(graph_randn_proj)
print(normalized_mutual_info_score(node_label,cluster_model.labels_))

#graph_randn_proj_2 = clf_proj.fit_transform(graph_randn_proj)

#clf.fit(graph_randn_proj_2,node_label)
#clf.score(graph_randn_proj_2,node_label)
'''



