from utils_new import *
from config import SAVE_DIR, VAEGConfig
from cell import VAEGCell
from rlcell import VAEGRLCell
import tensorflow as tf
import numpy as np
import logging
import copy
import os
import time
import networkx as nx
from collections import defaultdict
from operator import itemgetter
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VAEGRL(VAEGConfig):
    def __init__(self, hparams, placeholders, num_nodes, num_features, log_fact_k, input_size, istest=False):
        self.features_dim = num_features
        self.input_dim = num_nodes
        self.dropout = placeholders['dropout']
        self.k = hparams.random_walk
        self.lr = placeholders['lr']
        self.decay = placeholders['decay']
        self.n = num_nodes
        self.d = num_features
        self.z_dim = hparams.z_dim
        self.bin_dim = hparams.bin_dim
        self.mask_weight = hparams.mask_weight
        self.log_fact_k = log_fact_k
        self.neg_sample_size = hparams.neg_sample_size
        self.input_size = input_size
        self.combination = hparams.node_sample * hparams.bfs_sample
        self.temperature = hparams.temperature
        self.E = hparams.E

        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.features1 = tf.placeholder(dtype=tf.int32, shape=[self.n], name='features1')
        self.weight = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name="weight")
        self.weight_bin1 = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n, hparams.bin_dim], name="weight_bin1")
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.combination, None, hparams.bin_dim], name="weight_bin")
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')
        #self.neg_index = tf.placeholder(dtype=tf.int32,shape=[None], name='neg_index')
        self.edges = tf.placeholder(dtype=tf.int32, shape=[self.combination, None, 2], name='edges') 
        self.all_edges = tf.placeholder(dtype=tf.int32, shape=[self.combination, None, 2], name='all_edges')
        # for the time being 5 trajectories are in action
        self.trajectories = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name="trajectories")
        self.properties = tf.placeholder(dtype=tf.float32, name="properties")
        self.importance_weight = tf.placeholder(dtype=tf.float32, name="importance_weight")
        self.n_fill_edges = tf.placeholder(dtype=tf.int32)
        #self.known_edges = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='known_edges') 

        #node_count = [len(edge_list) for edge_list in self.edges]
        print("Debug Input size", self.input_size)
        node_count_tf = tf.fill([1, self.input_size],tf.cast(self.n, tf.float32))
        node_count_tf = tf.Print(node_count_tf, [node_count_tf], message="My node_count_tf")
        print("Debug size node_count", node_count_tf.get_shape())
        
        #tf.convert_to_tensor(node_count, dtype=tf.int32)
        self.cell = VAEGCell(self.adj, self.weight, self.features, self.z_dim, self.bin_dim, tf.to_float(node_count_tf), self.all_edges)
        self.c_x, enc_mu, enc_sigma, self.debug_sigma, dec_out, prior_mu, prior_sigma, z_encoded, w_edge, label, lambda_n, lambda_e = self.cell.call(self.input_data, self.n, self.d, self.k, self.combination, self.eps, hparams.sample)
        self.prob = dec_out
        #print('Debug', dec_out.shape)
        self.z_encoded = z_encoded
        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.w_edge = w_edge
        self.label = label
        self.lambda_n = lambda_n
        self.lambda_e = lambda_e

        #adj, weight, features, z_dim, bin_dim, node_count, edges, enc_mu, enc_sigma
        self.rlcell = VAEGRLCell(self.adj, self.weight, self.features, self.z_dim, self.bin_dim, self.all_edges, enc_mu, enc_sigma)
        #self, adj, weight, features, z_dim, bin_dim, enc_mu, enc_sigma, edges, index
        self.rl_dec_out, self.rl_w_edge = self.rlcell.call(self.input_data, self.n, self.d, self.k, self.combination, self.eps, hparams.sample)
        print_vars("trainable_variables")
        total_cost = 0.0
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        ll = []
        #self.grad = []
	self.grad_placeholder = []
        ll_rl = []
        self.apply_transform_op = []
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'RL' in var.name]
        for x in range(len(g_vars)):
            self.grad_placeholder.append(tf.placeholder(dtype=tf.float32, shape=g_vars[x].get_shape().as_list()))
	    #self.grad.append(tf.placeholder(dtype=tf.float32, shape=g_vars[x].get_shape().as_list()))
        print("Debug gvars", g_vars)
        V = []
        ll = self.likelihood(self.trajectories, self.prob[0], self.w_edge[0])
        ll_rl = self.likelihood(self.trajectories, self.rl_dec_out[0], self.rl_w_edge[0])
        w = tf.multiply(tf.divide(ll, ll_rl), tf.exp(-tf.divide(self.properties,self.temperature)))
        #print("Debug w")
        self.importance_weight = self.importance_weight + w
        self.loss = self.temperature * tf.log(tf.divide(ll_rl, ll)) + self.properties
        #tf.add(ll_rl, 1e-09)
        #print("Debug ll_rl")
        grad = self.train_op.compute_gradients(tf.log(tf.add(ll_rl, 1e-09)), var_list=g_vars)
        temp_grad = []
        temp_grad_val = []
        for x in range(len(g_vars)):
            #print("Debug grad", grad[x])
	    if grad[x][0] is not None:
                    g = grad[x]
                    #g = tf.multiply(grad[x][0], importance_weight)
            else:
                    #print("Debug None")
                    g = (tf.fill(tf.shape(g_vars[x]), 0.0), grad[x][1])
            #y = tf.add(self.grad[x], w * g[x][0])
            #print("Debug add")
            #temp_grad.append((tf.add(self.grad[x], w * g[0]), g[1]))
	    temp_grad.append((tf.add(self.grad_placeholder[x], w * g[0]), g[1]))
            temp_grad_val.append(tf.add(self.grad_placeholder[x], w * g[0]))
            print("Debug completed")
            #self.grad[x] = temp_grad[x][0]
        
        print("Debug temp grad", temp_grad)
        self.apply_transform_op = self.train_op.apply_gradients(temp_grad)
        self.grad = temp_grad_val

        '''
        for i in range(5):
            ll.append(self.likelihood(self.trajectories[i], self.prob[0], self.w_edge[0]))
            ll_rl.append(self.likelihood(self.trajectories[i], self.rl_dec_out[0], self.rl_w_edge[0]))
            V.append(tf.divide(ll[i], ll_rl[i]) * tf.exp(-tf.divide(self.properties[i],self.temperature)))
        V_total = tf.reduce_sum(V)
        for i in range(5):
            #total_cost += cost
            #importance_weight = tf.divide(tf.exp(-V[i]) * (1/self.temperature), V_total)
            importance_weight = tf.divide(V[i], V_total)
            grad = self.train_op.compute_gradients(tf.log(tf.add(ll_rl[i], 1e-09)), var_list=g_vars)
            #grad_new = [(grad, var) if grad is not None else tf.zeros_like(var) for  grad, var in zip(grads, var_list)]
            print("Debug importance_weight", importance_weight, grad[0], )
            for x in range(len(grad)):
                if grad[x][0] is not None:
                    g = grad[x]
                    #g = tf.multiply(grad[x][0], importance_weight)
                else:
                    g = (tf.fill(tf.shape(g_vars[x]), 0.0), grad[x][1])
                print("Debug g", g)

                if len(self.grad) > x:
                    self.grad[x] = (tf.add(self.grad[x][0], tf.multiply(g[0], importance_weight)/5), g[1])
                else:
                    self.grad.append(g)
            print("Debug grad", self.grad)
            self.apply_transform_op.append(self.train_op.apply_gradients(self.grad))
        '''
        self.sess = tf.Session()
        # We are considering 10 trajectories only

    def likelihood(self, edge_list, prob_dict, w_edge):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            k = 0
            with tf.variable_scope('NLL'):
                dec_mat_temp = tf.reshape(prob_dict, [self.n, self.n])

                #w_edge_exp = tf.exp(tf.minimum(tf.reshape(w_edge, [self.n, self.n, self.bin_dim]), tf.fill([self.n, self.n, self.bin_dim], 10.0)))
                #w_edge_pos = tf.multiply(self.weight_bin1, w_edge_exp)
                
                #w_edge_total = tf.reduce_sum(w_edge_exp, axis=1)
                #w_edge_score = tf.divide(w_edge_pos, w_edge_total)
                
                dec_mat = tf.exp(tf.minimum(dec_mat_temp, tf.fill([self.n, self.n], 10.0)))
                dec_mat = tf.Print(dec_mat, [dec_mat], message="my decscore values:")

                print "Debug dec_mat", dec_mat.shape, dec_mat.dtype, dec_mat
                comp = tf.subtract(tf.ones([self.n, self.n], tf.float32), self.adj)
                comp = tf.Print(comp, [comp], message="my comp values:")

                temp = tf.reduce_sum(tf.multiply(comp,dec_mat))
                negscore = tf.fill([self.n,self.n], temp+1e-9)
                negscore = tf.Print(negscore, [negscore], message="my negscore values:")

                posscore = tf.multiply(edge_list, dec_mat)
                posscore = tf.Print(posscore, [posscore], message="my posscore values:")

                #dec_out = tf.multiply(self.adj, dec_mat) 
                softmax_out = tf.divide(posscore, tf.add(posscore, negscore))
                ll = tf.reduce_sum(tf.log(tf.add(softmax_out, tf.fill([self.n,self.n], 1e-9))))
                ll= tf.Print(ll, [ll], message="My loss")
            return (ll)

    def get_trajectories_synthetic(self, p_theta, w_theta, edges, n_fill_edges):

            list_edges = []
            prob = np.reshape(p_theta,(self.n, self.n))
            temp = np.ones([self.n, self.n])
            p_rs = np.exp(np.minimum(prob, 10 * temp))
            denom = np.sum(p_rs)
            p = np.divide(p_rs, denom)
            p_new = []
        
            for i in range(self.n):
                for j in range(i+1, self.n):
                    list_edges.append((i,j,1))
                    p_new.append(p[i][j])

            print("Debug list edges", len(list_edges))
            known_edges = []
            for k in range(self.E):
                (u,v) = edges[k]
                if u < v:
                    list_edges.remove((u, v, 1))
                    p_new.remove(p[u][v])
                    known_edges.append((u, v, 1))
                else:
                    list_edges.remove((v, u, 1))
                    p_new.remove(p[v][u])
                    known_edges.append((v, u, 1))

            p = p_new / np.array(p_new).sum()
            
            print("Debug list_edges", len(list_edges), n_fill_edges) 
            trial = 0
            candidate_edges = []
            G = nx.Graph()
            #for j in range(10):
            adj = np.zeros((self.n, self.n))
            while trial < 500:
                    candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[n_fill_edges], p=p, replace=False)]
                    candidate_edges.extend(known_edges)
                    
                    candidate_edges_weighted = []
                    for (u, v, w) in candidate_edges:

                    	if u < v:
                        	candidate_edges_weighted.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
                    	else:
                        	candidate_edges_weighted.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")

                    G = nx.parse_edgelist(candidate_edges_weighted, nodetype=int)
                    if nx.is_connected(G):
                        for (u, v, w) in candidate_edges:
                            adj[u][v] = w
                            adj[v][u] = w
                        print("Debug trial", trial, len(G.edges()))
                        break
                    trial += 1
                    print("Trial", trial)
            return adj, G

    def get_trajectories(self, p_theta, w_theta, edges, weight, n_fill_edges, atom_list):

            indicator = np.ones([self.n, self.bin_dim])
            edge_mask = np.ones([self.n, self.n])
            degree = np.zeros(self.n)
            #print("Debug known edges", tf.shape(self.known_edges),self.known_edges.get_shape())
            #N = tf.stack([tf.shape(self.known_edges)[0]])[0]
            #known_edges = tf.unstack(self.known_edges)
            # For the time being make the number of known edges a constant E
            #'''
            known_edges = []
            for k in range(self.E):
                (u,v) = edges[k]
                edge_mask[u][v] = 0
                edge_mask[v][u] = 0
                degree[u]+=weight[u][v]
                degree[v]+=weight[v][u]
                known_edges.append((u,v,weight[u][v]))
                if (4 - degree[u]) == 0:
                    indicator[u][0] = 0
                if (4 - degree[u]) <= 1:
                    indicator[u][1] = 0
                if (4 - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (4 - degree[v]) == 0:
                    indicator[v][0] = 0
                if (4 - degree[v]) <= 1:
                    indicator[v][1] = 0
                if (4 - degree[v]) <= 2:
                    indicator[v][2] = 0
            #'''
            trial = 0
            candidate_edges = []
            G = nx.Graph()

            while trial < 50:

                #candidate_edges = 
                #candidate_edges = 
                #self.get_masked_candidate_with_atom_ratio_new(p_theta, w_theta, node_list, self.n_fill_edges, 1)
                #get_weighted_edges(indicator, p_theta, edge_mask, w_theta, self.n_fill_edges, node_list, degree)
                candidate_edges = get_masked_candidate_new(p_theta, w_theta, n_fill_edges, atom_list, indicator, edge_mask, degree)
                candidate_edges.extend(known_edges)
                G = nx.Graph()
                G.add_nodes_from(range(self.n))
                G.add_weighted_edges_from(candidate_edges)
                if nx.is_connected(G):
                    print("Debug trial", trial)
                    break
                trial += 1
                print("Trial", trial)
            return candidate_edges, G

    

    def initialize(self):
        logger.info("Initialization of parameters")
        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt == None or ckpt.model_checkpoint_path == None:
            self.initialize()
        else:
            print("Load the model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def copy_weight(self, copydir):
        self.initialize()
        print("Debug all", tf.global_variables())
        var_old = [v for v in tf.global_variables() if "RL" not in v.name]
        print("Debug var_old", var_old)
        saver = tf.train.Saver(var_old)
        ckpt = tf.train.get_checkpoint_state(copydir)
        
        #print_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path, tensor_name='', all_tensors='')
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        var_new = [v for v in tf.global_variables() if "RL" in v.name]
        for v in var_new:
            v_old = [v_old for v_old in tf.global_variables() if v_old.name == v.name.replace("RL", "")][0]
            print("v_old", v_old.value(), v_old.name) 
            #if v_old  in var_old
            assign = tf.assign(v, v_old)
	    self.sess.run(assign)
	    #v = tf.Variable(v.name.replace("RL", ""))
	    print("v_new", v, v.name)
    
    def train(self, placeholders, hparams, adj, weight, weight_bin, weight_bin1, features, edges, all_edges, features1, atom_list):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate
        decay = hparams.decay_rate

        f1 = open(hparams.out_dir + '/iteration.txt', 'r')
        iteration = int(f1.read().strip())

        # training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)
        start_before_epoch = time.time()
        importance_weight = 0.0
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'RL' in var.name]
        print("Debug g_vars", g_vars)
        grad_local = []
        for x in range(len(g_vars)):
            a = np.zeros(shape=g_vars[x].get_shape().as_list(), dtype=float)
            #a.fill(0.0)
            print("Debug a", a, a.shape)
            grad_local.append(a)
        print("Debug gradlocal", grad_local, g_vars[0].get_shape().as_list())
        #'''
        for i in range(len(adj)):
            for epoch in range(num_epochs):
            	start = time.time()
            	#for i in range(len(adj)):
                if len(edges[i]) == 0:
                    continue
                # Learning rate decay
                #self.sess.run(tf.assign(self.lr, self.lr * (self.decay ** epoch)))
                start1 = time.time()
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                feed_dict.update({self.adj: adj[i]})
                eps = np.random.randn(self.n, self.z_dim, 1)  
                #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
                
                feed_dict.update({self.features: features[i]})
                feed_dict.update({self.features1: features1[i]})
                feed_dict.update({self.weight_bin: weight_bin[i]})
                feed_dict.update({self.weight_bin1: weight_bin1[i]})
                feed_dict.update({self.weight: weight[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                feed_dict.update({self.n_fill_edges: len(edges[i][0])-self.E })               
                feed_dict.update({self.edges:edges[i]})
                feed_dict.update({self.all_edges:[all_edges[i]]})
                feed_dict.update({self.trajectories: adj[i]})
                feed_dict.update({self.properties:5.0})
                
                #input_, train_loss, _, probdict, cx, w_edge, lambda_e, lambda_n = self.sess.run([self.input_data ,self.cost, self.apply_transform_op, self.prob, self.c_x, self.w_edge, self.lambda_e, self.lambda_n], feed_dict=feed_dict)
                prob, w_edge, rl_prob, rl_w_edge, lambda_e, lambda_n,z_encoded = self.sess.run([self.prob, self.w_edge, self.rl_dec_out, self.rl_w_edge, self.lambda_e, self.lambda_n, self.z_encoded], feed_dict=feed_dict)
                print("Debug shapes" , rl_prob[0][0], prob[0][0], z_encoded )
                end1 = time.time()
                trajectories = []
                properties = []
                
                for j in range(1):
                    t, G = self.get_trajectories_synthetic(rl_prob, rl_w_edge, edges[i][0], len(edges[i][0])-self.E)
                    print("Debug properties",compute_cost_dia(G))
                    feed_dict.update({self.trajectories: t})
                    feed_dict.update({self.properties:compute_cost_dia(G)})
                    feed_dict.update({self.importance_weight:importance_weight})
                    #feed_dict.update({i: d for i, d in zip(self.grad, grad_local)})
		    feed_dict.update({i: d for i, d in zip(self.grad_placeholder, grad_local)})
                    #grad_local, importance_weight = self.sess.run([self.grad, self.importance_weight], feed_dict=feed_dict)
                    _, grad_local, importance_weight, train_loss = self.sess.run([self.apply_transform_op, self.grad, self.importance_weight, self.loss], feed_dict=feed_dict)
                    print("Debug grad_local", grad_local)
                    print("Time size of graph", len(tf.get_default_graph().get_operations()))
                    end2 = time.time()
                    #train_loss = self.sess.run([self.cost], feed_dict=feed_dict)
                    #input_, train_loss, _, probdict, cx, w_edge, lambda_e, lambda_n= self.sess.run([self.input_data ,self.cost, self.apply_transform_op, self.prob, self.c_x, self.w_edge, self.lambda_e, self.lambda_n], feed_dict=feed_dict)
                
                    iteration += 1
                    #print("Lambda_e, lambda_n", lambda_e, lambda_n, i)
                    if iteration % hparams.log_every == 0 and iteration > 0:
                        #print(train_loss)
                        print("{}/{}(epoch {}), train_loss = {:.6f}".format(iteration, num_epochs, epoch + 1, train_loss))
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        saver.save(self.sess, checkpoint_path, global_step=iteration)
                        logger.info("model saved to {}".format(checkpoint_path))
                end3 = time.time()
                print("Time debug", end1 - start1, end2 - end1, end3 - end2, end2 - start1)
            end = time.time()
            print("Time taken for a batch: ",end - start, end2 - start1)
	    #print("Time size of graph", len(tf.get_default_graph().get_operations()))
	    #operations = tf.get_default_graph().get_operations()
	    #print("Operations:")
	    #for op in operations:
        #    print("OP",op)
	    #print("operations", tf.get_default_graph().get_operations())
        end_after_epoch = time.time()
        print("Time taken to completed all epochs", -start_before_epoch + end_after_epoch)
        f1 = open(hparams.out_dir+'/iteration.txt','w')
        f1.write(str(iteration))
        #'''

    def getembeddings(self, hparams, placeholders, adj, deg, weight_bin, weight):

        eps = np.random.randn(self.n, self.z_dim, 1)

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                        hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
        feed_dict.update({self.features: deg})
        feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
        feed_dict.update({self.eps: eps})
        feed_dict.update({self.weight_bin: weight_bin})
        feed_dict.update({self.weight: weight})

        prob, ll, kl, w_edge, embedding = self.sess.run([self.prob, self.ll, self.kl, self.w_edge, self.z_encoded],
                                                        feed_dict=feed_dict)
        return embedding

    def get_masked_candidate_with_atom_ratio_new(self, prob, w_edge, atom_count, num_edges, hde):
        rest = range(self.n)
        nodes = []
        hn = []
        on = []
        nn = []
        cn = []

        for i in range(self.n):
            if atom_count[i] == 1:
                hn.append(i)
            if atom_count[i] == 2:
                on.append(i)
            if atom_count[i] == 3 or atom_count[i] == 5:
                nn.append(i)
            if atom_count[i] == 4:
                cn.append(i)


        nodes.extend(hn)
        nodes.extend(cn)
        nodes.extend(on)
        nodes.extend(nn)

        node_list = atom_count
        print("Debug nodelist", node_list)
        
        indicator = np.ones([self.n, self.bin_dim])
        edge_mask = np.ones([self.n, self.n])
        degree = np.zeros(self.n)

        for node in hn:
            indicator[node][1] = 0
            indicator[node][2] = 0
        for node in on:
            indicator[node][2] = 0

        # two hydrogen atom cannot have an edge between them
        for n1 in hn:
            for n2 in hn:
                edge_mask[n1][n2] = 0
        candidate_edges = []
        # first generate edges joining with Hydrogen atoms sequentially
        index = 0
        i = 0
        hydro_sat = np.zeros(self.n)
        #first handle hydro
        try:
         for node in nodes:
            deg_req = node_list[node]
            d = degree[node]
            list_edges = get_candidate_neighbor_edges(node, self.n)
            if node in hn:
                for i1 in range(self.n):
                    if hydro_sat[i1] == node_list[i1] - 1:
                        edge_mask[i1][node] = 0
                        edge_mask[node][i1] = 0
            while d < deg_req:
                p = normalise_h1(prob, w_edge,  self.bin_dim, indicator, edge_mask, node)
                
                candidate_edges.extend([list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])

                (u, v, w) = candidate_edges[i]
                degree[u]+= w
                degree[v]+= w
                d += w
                if u in hn:
                    hydro_sat[v] += 1
                if v in hn:
                    hydro_sat[u] += 1
                edge_mask[u][v] = 0
                edge_mask[v][u] = 0
                
                if (node_list[u] - degree[u]) == 0 :
                    indicator[u][0] = 0
                if (node_list[u] - degree[u]) <= 1 :
                    indicator[u][1] = 0
                if (node_list[u] - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (node_list[v] - degree[v]) == 0 :
                    indicator[v][0] = 0
                if (node_list[v] - degree[v]) <= 1 :
                    indicator[v][1] = 0
                if (node_list[v] - degree[v]) <= 2:
                    indicator[v][2] = 0
                
                i+=1 
                print("Debug candidate_edges", candidate_edges[i - 1])
                #    print("change state", el, degree[el], node_list[el], indicator[el])
                #'''
        except:
            if len(candidate_edges) < 1:
                candidate_edges = []
        candidate_edges_new = []
        for (u, v, w) in candidate_edges:
            if u < v:
                candidate_edges_new.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
            else:
                candidate_edges_new.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")
        print("Candidate_edges_new", candidate_edges_new)
        return candidate_edges_new


    def get_masked_candidate(self, list_edges, prob, w_edge, num_edges, hde, indicator=[], degree=[]):

        list_edges_original = copy.copy(list_edges)
        n = len(prob[0])
        # sample 1000 times
        count = 0
        structure_list = defaultdict(int)

        # while(count < 50):
        while (count < 1):
            applyrules = False
            list_edges = copy.copy(list_edges_original)
            if len(indicator) == 0:
                print("Debug indi new assign")
                indicator = np.ones([self.n, self.bin_dim])
            reach = np.ones([n, n])

            p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [1], p=p, replace=False)]
            # if degree == None:
            if len(degree) == 0:
                print("Debug degree new assign")
                degree = np.zeros([self.n])
            G = None
            saturation = 0

            for i1 in range(num_edges - 1):
                (u, v, w) = candidate_edges[i1]
                for j in range(n):

                    if reach[u][j] == 0:
                        reach[v][j] = 0
                        reach[j][v] = 0
                    if reach[v][j] == 0:
                        reach[u][j] = 0
                        reach[j][u] = 0

                reach[u][v] = 0
                reach[v][u] = 0

                degree[u] += w
                degree[v] += w

                if degree[u] >= 4:
                    indicator[u][0] = 0
                if degree[u] >= 3:
                    indicator[u][1] = 0
                if degree[u] >= 2:
                    indicator[u][2] = 0

                if degree[v] >= 4:
                    indicator[v][0] = 0
                if degree[v] >= 3:
                    indicator[v][1] = 0
                if degree[v] >= 2:
                    indicator[v][2] = 0

                # there will ne bo bridge
                p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, candidate_edges, list_edges, indicator)

                try:
                    candidate_edges.extend([list_edges[k] for k in
                                            np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                except:
                    # candidate_edges = []
                    continue
            structure_list[
                ' '.join([str(u) + '-' + str(v) + '-' + str(w) for (u, v, w) in sorted(candidate_edges)])] += 1
            count += 1

        # return the element which has been sampled maximum time
        return max(structure_list.iteritems(), key=itemgetter(1))[0]

    def get_unmasked_candidate(self, list_edges, prob, w_edge, num_edges):
        # sample 1000 times
        count = 0
        structure_list = defaultdict(int)

        # while (count < 1000):
        while (count < 50):
            indicator = np.ones([self.n, self.bin_dim])
            p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [num_edges], p=p, replace=False)]
            structure_list[' '.join([str(u) + '-' + str(v) + '-' + str(w) for (u, v, w) in
                                     sorted(candidate_edges, key=itemgetter(0))])] += 1

            # structure_list[sorted(candidate_edges, key=itemgetter(1))] += 1
            count += 1

        # return the element which has been sampled maximum time
        return max(structure_list.iteritems(), key=itemgetter(1))[0]

    def sample_graph_posterior_new(self, hparams, placeholders, adj, features, weight_bins, weights, embeddings, k=0):
        list_edges = get_candidate_edges(self.n)
        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                        hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
        feed_dict.update({self.features: features})
        feed_dict.update({self.weight_bin: weight_bins})
        feed_dict.update({self.weight: weights})
        feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
        feed_dict.update({self.eps: embeddings})
        hparams.sample = True

        prob, ll, z_encoded, enc_mu, enc_sigma, elbo, w_edge, labels = self.sess.run(
            [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge, self.label],
            feed_dict=feed_dict)
        # prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
        prob = np.reshape(prob, (self.n, self.n))

        w_edge = np.reshape(w_edge, (self.n, self.n, self.bin_dim))

        atom_list = [4, 4, 2, 4, 4, 3, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # self.getatoms(atom_list)
        if not hparams.mask_weight:
            candidate_edges = self.get_unmasked_candidate(list_edges, prob, w_edge, hparams.edges)
        else:
            i = 0
            hde = 1
            # while (i < 1000):
            candidate_edges = self.get_masked_candidate_with_atom_ratio_new(prob, w_edge, atom_list, hparams.edges, hde)
            # if len(candidate_edges) > 0:
            #        break
            #    i += 1

            # candidate_edges = self.get_masked_candidate(list_edges, prob, w_edge, hparams.edges, hde)
        with open(hparams.sample_file + 'temp.txt' + str(k), 'w') as f:
            for uvw in candidate_edges.split():
                [u, v, w] = uvw.split("-")
                u = int(u)
                v = int(v)
                w = int(w)
                if (u >= 0 and v >= 0):
                    # with open(hparams.sample_file + 'temp.txt', 'a') as f:
                    f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')

    def getatoms(self, node, label):
        label_new = np.reshape(label, (node, self.d))
        print("Debug label original shape:", label_new)

        label_new = np.exp(label_new)
        s = label_new.shape[0]
        print("Debug label shape:", label_new.shape, s)

        label_new_sum = np.reshape(np.sum(label_new, axis=1), (s, 1))
        print("Debug label sum:", label_new_sum.shape)

        prob_label = label_new / label_new_sum
        pred_label = np.zeros(4)
        valency_arr = np.zeros(node)

        print("Debug prob label shape:", prob_label.shape, prob_label)

        # print("Debug label", label_new)
        for i in range(node):
            valency = np.random.choice(range(4), [1], p=prob_label[i])
            pred_label[valency] += 1
            valency_arr[i] = valency + 1

        print("Debug pred_label", pred_label, valency_arr)
        return (pred_label, valency_arr)

    def sample_graph_neighborhood(self, hparams, placeholders, adj, features, weights, weight_bins, s_num, node, ratio,
                                  hde, num=10, outdir=None):
        list_edges = get_candidate_edges(self.n)

        # eps = load_embeddings(hparams.z_dir+'encoded_input0'+'.txt', hparams.z_dim)
        eps = np.random.randn(self.n, self.z_dim, 1)

        train_mu = []
        train_sigma = []
        hparams.sample = False

        # approach 1
        for i in range(len(adj)):
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                            hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
            feed_dict.update({self.features: features[i]})
            feed_dict.update({self.weight_bin: weight_bins[i]})
            feed_dict.update({self.weight: weights[i]})
            feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
            feed_dict.update({self.eps: eps})
            hparams.sample = False
            prob, ll, z_encoded, enc_mu, enc_sigma, elbo, w_edge = self.sess.run(
                [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],
                feed_dict=feed_dict)

            with open(hparams.z_dir + 'encoded_input' + str(i) + '.txt', 'a') as f:
                for z_i in z_encoded:
                    f.write('[' + ','.join([str(el[0]) for el in z_i]) + ']\n')
                f.write("\n")

            with open(hparams.z_dir + 'encoded_mu' + str(i) + '.txt', 'a') as f:
                for z_i in enc_mu:
                    f.write('[' + ','.join([str(el[0]) for el in z_i]) + ']\n')
                f.write("\n")

            with open(hparams.z_dir + 'encoded_sigma' + str(i) + '.txt', 'a') as f:
                for x in range(self.n):
                    for z_i in enc_sigma[x]:
                        f.write('[' + ','.join([str(el) for el in z_i]) + ']\n')
                    f.write("\n")

            hparams.sample = True

            # for j in range(self.n):
            # for j in [1, 5, 15]:
            for j in [1]:
                z_encoded_neighborhood = copy.copy(z_encoded)
                feed_dict.update({self.eps: z_encoded_neighborhood})
                prob, ll, z_encoded_neighborhood, enc_mu, enc_sigma, elbo, w_edge, labels = self.sess.run(
                    [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge,
                     self.label],
                    feed_dict=feed_dict)
                # prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
                with open(hparams.z_dir + 'sampled_z' + str(i) + '.txt', 'a') as f:
                    for z_i in z_encoded:
                        f.write('[' + ','.join([str(el[0]) for el in z_i]) + ']\n')
                    f.write("\n")

                prob = np.reshape(prob, (self.n, self.n))
                w_edge = np.reshape(w_edge, (self.n, self.n, self.bin_dim))
                with open(hparams.z_dir + 'prob_mat' + str(i) + '.txt', 'a') as f:
                    for x in range(self.n):
                        f.write('[' + ','.join([str(el) for el in prob[x]]) + ']\n')
                    f.write("\n")
                with open(hparams.z_dir + 'weight_mat' + str(i) + '.txt', 'a') as f:
                    for x in range(self.n):
                        f.write('[' + ','.join(
                            [str(el[0]) + ' ' + str(el[1]) + ' ' + str(el[2]) for el in w_edge[x]]) + ']\n')
                    f.write("\n")

                if not hparams.mask_weight:
                    print("Non mask")
                    candidate_edges = self.get_unmasked_candidate(list_edges, prob, w_edge, hparams.edges)
                else:
                    print("Mask")
                    (atom_list, valency_arr) = self.getatoms(hparams.nodes, labels)
                    candidate_edges = self.get_masked_candidate_with_atom_ratio_new(prob, w_edge, valency_arr,
                                                                                    hparams.edges, hde)

                for uvw in candidate_edges.split():
                    [u, v, w] = uvw.split("-")
                    u = int(u)
                    v = int(v)
                    w = int(w)
                    if (u >= 0 and v >= 0):
                        with open(hparams.sample_file + "approach_1_node_" + str(j) + "_" + str(s_num) + '.txt',
                                  'a') as f:
                            f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')

    def sample_graph_synthetic(self, hparams, placeholders, adj, weight, weight_bin, weight_bin1, features, edges, all_edges, features1, atom_list, k=0, outdir=None):
        '''
        Args :
            num - int
                10
                number of edges to be sampled
            outdir - string
            output dir
        '''
        list_edges = []

        for i in range(self.n):
            for j in range(i+1, self.n):
                    list_edges.append((i,j,1))

        hparams.sample=True

        eps = np.random.randn(self.n, self.z_dim, 1)
        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj[0]})

        feed_dict.update({self.features: features[0]})
        feed_dict.update({self.features1: features1[0]})
        feed_dict.update({self.weight_bin: weight_bin[0]})
        feed_dict.update({self.weight_bin1: weight_bin1[0]})
        feed_dict.update({self.weight: weight[0]})
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})
        feed_dict.update({self.n_fill_edges: len(edges[0][0])-self.E })
        feed_dict.update({self.edges:edges[0]})
        feed_dict.update({self.all_edges:[all_edges[0]]})

        prob, z_encoded, sample_mu, sample_sigma, w_edge = self.sess.run([self.rl_dec_out, self.z_encoded, self.enc_mu, self.enc_sigma, self.rl_w_edge],feed_dict=feed_dict )
        prob = np.reshape(prob,(self.n, self.n))
        temp = np.ones([self.n, self.n])
        p_rs = np.exp(np.minimum(prob, 10 * temp))
        denom = np.sum(p_rs, axis = 1)
        p = np.divide(p_rs, denom)
        p_new = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                p_new.append(p[i][j])
        p = p_new / np.array(p_new).sum()
        #w_edge = np.reshape(w_edge,(self.n, self.n, self.bin_dim))
        smiles = []
        trial = 0

        if not hparams.mask_weight:
            trial = 0
            while trial < 500:
                #print("Debug", len(list_edges))
                #test =  np.random.choice(range(len(list_edges)),[hparams.edges], p=p, replace=False)
                candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[100], p=p, replace=False)]
                candidate_edges_weighted = []
                for (u, v, w) in candidate_edges:
                    if u < v:
                        candidate_edges_weighted.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
                    else:
                        candidate_edges_weighted.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")

                G = nx.parse_edgelist(candidate_edges_weighted, nodetype=int)
                if nx.is_connected(G):
                        for (u,v,w) in candidate_edges:
                            if (u >= 0 and v >= 0):
                                with open(hparams.sample_file + "approach_2_" + str(trial) +"_"+str(k)+ '.txt', 'a') as f:
                                    f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                        break
                trial+= 1
                
    def sample_graph(self, hparams, placeholders, adj, features, weights, weight_bins, s_num, node, hde, num=10,
                     outdir=None):
        '''
        Args :
            num - int
                10
                number of edges to be sampled
            outdir - string
            output dir
        '''
        list_edges = []

        for i in range(self.n):
            for j in range(i + 1, self.n):
                list_edges.append((i, j, 1))
                list_edges.append((i, j, 2))
                list_edges.append((i, j, 3))
        # list_edges.append((-1, -1, 0))

        list_weight = [1, 2, 3]

        hparams.sample = True

        eps = np.random.randn(self.n, self.z_dim, 1)
        with open(hparams.z_dir + 'test_prior_' + str(s_num) + '.txt', 'a') as f:
            for z_i in eps:
                f.write('[' + ','.join([str(el[0]) for el in z_i]) + ']\n')

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                        hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj[0]})
        feed_dict.update({self.features: features[0]})
        feed_dict.update({self.weight_bin: weight_bins[0]})
        feed_dict.update({self.weight: weights[0]})

        feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
        feed_dict.update({self.eps: eps})

        prob, z_encoded, kl, sample_mu, sample_sigma, loss, w_edge, labels = self.sess.run(
            [self.prob, self.z_encoded, self.kl, self.enc_mu, self.enc_sigma, self.cost, self.w_edge,
             self.label], feed_dict=feed_dict)
        prob = np.reshape(prob, (self.n, self.n))
        w_edge = np.reshape(w_edge, (self.n, self.n, self.bin_dim))

        indicator = np.ones([self.n, 3])
        p, list_edges, w_new = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)

        if not hparams.mask_weight:
            trial = 0
            while trial < 5000:
                candidate_edges = [list_edges[i] for i in
                                   np.random.choice(range(len(list_edges)), [hparams.edges], p=p, replace=False)]
                with open(hparams.sample_file + 'test.txt', 'w') as f:
                    for (u, v, w) in candidate_edges:
                        if (u >= 0 and v >= 0):
                            f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                f = open(hparams.sample_file + 'test.txt')
                G = nx.read_edgelist(f, nodetype=int)
                if nx.is_connected(G):
                    for (u, v, w) in candidate_edges:
                        if (u >= 0 and v >= 0):
                            with open(hparams.sample_file + "approach_2_" + str(trial) + "_" + str(s_num) + '.txt',
                                      'a') as f:
                                f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                trial += 1

        else:
            trial = 0
            while trial < 5000:
                candidate_edges = self.get_masked_candidate(list_edges, prob, w_edge, hparams.edges, hde)
                # print("Debug candidate", candidate_edges)
                if len(candidate_edges) > 0:
                    with open(hparams.sample_file + 'test.txt', 'w') as f:
                        for uvw in candidate_edges.split():
                            [u, v, w] = uvw.split("-")
                            u = int(u)
                            v = int(v)
                            w = int(w)
                            if (u >= 0 and v >= 0):
                                f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                    f = open(hparams.sample_file + 'test.txt')
                    # try:
                    G = nx.read_edgelist(f, nodetype=int)
                    # except:
                    # continue

                    if nx.is_connected(G):
                        for uvw in candidate_edges.split():
                            [u, v, w] = uvw.split("-")
                            u = int(u)
                            v = int(v)
                            w = int(w)
                            if (u >= 0 and v >= 0):
                                with open(hparams.sample_file + "approach_2_" + str(trial) + "_" + str(s_num) + '.txt',
                                          'a') as f:
                                    f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                trial += 1
