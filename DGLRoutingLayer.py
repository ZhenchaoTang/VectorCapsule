import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl


class DGLRoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, batch_size=0, device='cpu'):
        super(DGLRoutingLayer, self).__init__()
        self.batch_size = batch_size
        self.g = init_graph(in_nodes, out_nodes, f_size, device=device)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
        self.device = device

    def forward(self, u_hat, routing_num=1):
        self.g.edata['u_hat'] = u_hat
        batch_size = self.batch_size

        # step 2 (line 5)
        def cap_message(edges):
            if batch_size:
                return {'m': edges.data['c'].unsqueeze(1) * edges.data['u_hat']}
            else:
                return {'m': edges.data['c'] * edges.data['u_hat']}

        def cap_reduce(nodes):
            return {'s': th.sum(nodes.mailbox['m'], dim=1)}

        for r in range(routing_num):
            # step 1 (line 4): normalize over out edges
            edges_b = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            self.g.edata['c'] = F.softmax(edges_b, dim=1).view(-1, 1)

            # Execute step 1 & 2
            self.g.update_all(message_func=cap_message, reduce_func=cap_reduce)

            # step 3 (line 6)
            if self.batch_size:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
            else:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            # step 4 (line 7)
            v = th.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            if self.batch_size:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
            else:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)


def squash(s, dim=1):
    sq = th.sum(s ** 2, dim=dim, keepdim=True)
    s_norm = th.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_norm)
    return s


def init_graph(in_nodes, out_nodes, f_size, device='cpu'):
    g = dgl.DGLGraph()
    g.set_n_initializer(dgl.frame.zero_initializer)
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)
    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)

    g = g.to(device)
    g.edata['b'] = th.zeros(in_nodes * out_nodes, 1).to(device)
    return g


if __name__=="__main__":
    # ToDo: monitoring the entropy of coupling coefficients
    import numpy as np
    import matplotlib.pyplot as plt

    in_nodes = 20
    out_nodes = 10
    f_size = 4
    u_hat = th.randn(in_nodes * out_nodes, f_size)
    routing = DGLRoutingLayer(in_nodes, out_nodes, f_size)

    entropy_list = []
    dist_list = []

    for i in range(10):
        routing(u_hat)
        dist_matrix = routing.g.edata['c'].view(in_nodes, out_nodes)
        entropy = (-dist_matrix * th.log(dist_matrix)).sum(dim=1)
        entropy_list.append(entropy.data.numpy())
        dist_list.append(dist_matrix.data.numpy())

    stds = np.std(entropy_list, axis=1)
    means = np.mean(entropy_list, axis=1)
    plt.figure()
    plt.errorbar(np.arange(len(entropy_list)), means, stds, marker='o')
    plt.ylabel("Entropy of Weight Distribution")
    plt.xlabel("Number of Routing")
    plt.xticks(np.arange(len(entropy_list)))
    plt.show()

    # ToDo: watching the evolution of histograms
    import seaborn as sns
    import matplotlib.animation as animation

    fig,ax = plt.subplots()
    def dist_animate(i):
        ax.cla()
        sns.distplot(dist_list[i].reshape(-1), kde=False, ax=ax)
        ax.set_xlabel("Weight Distribution Histogram")
        ax.set_title("Routing: %d" % (i))
    ani = animation.FuncAnimation(fig, dist_animate, frames=len(entropy_list), interval=500)
    plt.show()

    # ToDo: monitoring the how lower-level Capsules gradually attach to one of the higher level ones
    import networkx as nx
    from networkx.algorithms import bipartite

    g = routing.g.to_networkx()
    X, Y = bipartite.sets(g)
    height_in = 10
    height_out = height_in * 0.8
    height_in_y = np.linspace(0, height_in, in_nodes)
    height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
    pos = dict()

    fig2,ax2 = plt.subplots()
    pos.update((n, (i, 1)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
    pos.update((n, (i, 2)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2

    def weight_animate(i):
        ax2.cla()
        ax2.axis('off')
        ax2.set_title("Routing: %d  " % i)
        dm = dist_list[i]
        nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='r', node_size=100, ax=ax2)
        nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='b', node_size=100,
                               ax=ax2)
        for edge in g.edges():
            nx.draw_networkx_edges(g, pos, edgelist=[edge], width=dm[edge[0], edge[1] - in_nodes] * 1.5, ax=ax2)

    ani2 = animation.FuncAnimation(fig2, weight_animate, frames=len(dist_list), interval=500)
    plt.show()