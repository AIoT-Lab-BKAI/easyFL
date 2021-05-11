from task import modelfuncs
from .fedbase import BaseServer, BaseClient
import numpy as np
import copy
import cvxopt

def quadprog(P, q, G, h, A, b):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        # algorithm hyper-parameters
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['learning_rate']
        self.epsilon = option['epsilon']
        self.paras_name = ['epsilon']

    def iterate(self, t):
        ws, losses, grads = [], [], []
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
            grad_i = modelfuncs.modeldict_sub(self.model.state_dict(), w)
            # clip gi
            grad_i = modelfuncs.modeldict_scale(grad_i, 1.0 / modelfuncs.modeldict_norm(grad_i))
            grads.append(grad_i)
        # calculate λ0
        nks = [self.client_vols[cid] for cid in selected_clients]
        nt = sum(nks)
        lambda0 = [1.0*nk/nt for nk in nks]
        # optimize lambdas to minimize ||λ'g||² s.t. λ∈Δ, ||λ - λ0||∞ <= ε
        self.dynamic_lambdas = self.optim_lambda(grads, lambda0)
        self.dynamic_lambdas = [ele[0] for ele in self.dynamic_lambdas]
        # aggregate grads
        dt = self.aggregate(grads, self.dynamic_lambdas)
        # update model
        w_new = modelfuncs.modeldict_sub(self.model.state_dict(), modelfuncs.modeldict_scale(dt, self.learning_rate))
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def aggregate(self, ws, p=[]):
        return modelfuncs.modeldict_weighted_average(ws, p)

    def optim_lambda(self, grads, lambda0):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(modelfuncs.modeldict_to_tensor1D(gi)).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lambda0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lambda0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2*n,n))
        for i in range(n):
            G[i][i]=-1
            G[n+i][i]=1
        h = np.zeros((2*n,1))
        for i in range(n):
            h[i] = -lb[i]
            h[n+i] = ub[i]
        res=quadprog(P, q, G, h, A, b)
        return res

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)