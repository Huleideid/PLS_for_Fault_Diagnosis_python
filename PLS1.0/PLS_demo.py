# 导入相关库
import numpy as np
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt


def autos(X):
    m = X.shape[0]
    n = X.shape[1]
    X_m = np.zeros((m, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    for i in range(n):
        a = np.ones(m) * mu[i]
        X_m[:, i] = (X[:, i]-a) / sigma[i]
    return X_m, mu, sigma

def autos_test(data,m_train,v_train):
    m = data.shape[0]
    n = data.shape[1]
    data_new = np.zeros((m, n))
    for i in range(n):
        a = np.ones(m) * m_train[i]
        data_new[:, i] = (data[:, i] - a) / v_train[i]
    return data_new

def pls_nipals(X, Y, A, max_iter=2000, epsilon=1e-10):
    olv=A
    rankx=np.linalg.matrix_rank(X)
    if olv>=rankx:
        A=rankx
        
    ssqx=np.sum(X**2)
    ssqy=np.sum(Y**2)
    ssq=np.zeros((A,2))
    ssqdiff=np.zeros((A,2))
    t_old = 0
    iters = 0 
    u = Y[:,0].reshape(Y.shape[0],1)
    while iters < max_iter:
        W = X.T @ u / (np.linalg.norm(X.T @ u))
        W = W/np.linalg.norm(W)
        T = X @ W
        Q = Y.T @ T / (T.T @ T)
        Q=Q/np.linalg.norm(Q)
        u = Y @ Q
        t_diff = T - t_old
        t_old = T
        if np.linalg.norm(t_diff) < epsilon:
            P = X.T @ T / (T.T @ T)
            X = X - T @ (P.T)
            B=u.T@T/(T.T@T)
            Y=Y-B[0,0]*T@Q.T
            break
        else:
            iters += 1
            
    ssq[0,0] = np.sum(X**2)*100/ssqx;
    ssq[0,1] = np.sum(Y**2)*100/ssqy;
    
    for i in range(1,A):
        t_old = 0
        iters = 0
        u = Y[:,0].reshape(Y.shape[0],1)
        while iters < max_iter:
            w = X.T @ u / (np.linalg.norm(X.T @ u))
            w = w/np.linalg.norm(w)
            t = X @ w
            q = Y.T @ t / (t.T @ t)
            q=q/np.linalg.norm(q)
            u = Y @ q
            t_diff = t - t_old
            t_old = t
            if np.linalg.norm(t_diff) < epsilon:
                p = X.T @ t / (t.T @ t)
                p=p/np.linalg.norm(p)
                X = X - t @ (p.T)
                b=u.T@t/(t.T@t)
                Y=Y-b[0,0]*t@q.T
                t_old = t
                T = np.hstack((T,t))               
                W = np.hstack((W,w))
                Q = np.hstack((Q,q))
                P = np.hstack((P,p))
                B = np.hstack((B,b))
                break
            else:
                iters += 1    
        ssq[i,0] = np.sum(X**2)*100/ssqx;
        ssq[i,1] = np.sum(Y**2)*100/ssqy;

    ssqdiff[0,0] = 100 - ssq[0,0];
    ssqdiff[0,1] = 100 - ssq[0,1];
    ssqdiff[1:,:]=ssq[0:-1,:]-ssq[1:,:]
    R = W @ np.linalg.inv((P.T @ W))
    return T,W,Q,P,R,B,ssqdiff,ssq


path_train = './data/d00.mat'
path_test= './data/d01te.mat'
data1 = loadmat(path_train)['d00']
X1 = data1[:,:22]
X2 = data1[:,-11:]
X_Train= np.hstack((X1,X2))
Y_Train = data1[:,34:35]

data2 = loadmat(path_test)['d01te']
X11 = data2[:,:22]
X22 = data2[:,-11:]
X_test = np.hstack((X11,X22))
# Y_test  = data2[:,34:36]
Y_test  = data2[:,34:35]

# 数据标准化
##训练数据标准化
X_train,X_mean,X_s = autos(X_Train)
Y_train,Y_mean,Y_s = autos(Y_Train)
##测试数据标准化
X_test = autos_test(X_test,X_mean,X_s)
Y_test = autos_test(Y_test,Y_mean,Y_s)


A = 6#pls主元数
T, W, Q, P, R, B, ssqdiff, ssq = pls_nipals(X_train, Y_train, A)

## pls建模
alpha=0.95;# 显著性水平
X_hat = X_train-X_train@R@P.T
n = X_train.shape[0]
T2_lim = A*(n** 2-1)/(n*(n-A)) * scipy.stats.f.ppf(alpha, A, n-A) # T2控制限 

# 计算控制限
Qx_normal=[]
for i in range(X_hat.shape[0]):
    Qx_normal.append(X_hat[i,:].T @ X_hat[i,:])
S1=np.var(Qx_normal); 
mio1=np.mean(Qx_normal);
V1=2*mio1**2/S1; 
Q_lim=S1/(2*mio1)*scipy.stats.chi2.ppf(alpha,V1);

# 计算测试数据的监控结果
T_value = []
Q_value = []
for i in range(X_test.shape[0]):
    t = R.T @ X_test[i,:]
    xr_old = X_test[i,:].T-P@R.T@X_test[i,:].T
    T_value.append(t.T @ np.linalg.inv((T.T @ T) / (X_train.shape[0] - 1)) @ t)
    Q_value.append(xr_old.T@xr_old)

# 对测试数据的监控结果可视化
plt.figure()
plt.subplot(2,1,1)
plt.plot(T_value)
plt.xlabel('Sample number')
plt.ylabel('$T^2$')
plt.axhline(y=T2_lim,ls="--",color="r")
plt.subplot(2,1,2)
plt.plot(Q_value)
plt.xlabel('Sample number')
plt.ylabel('$Q$')
plt.axhline(y=Q_lim,ls="--",color="r")
plt.show()

# pls预测结果可视化
T_pred = X_train @ R;
Y_pred = T_pred @ np.diag(B[0])@ Q.T;

plt.figure()
plt.plot(Y_pred,color="r")
plt.plot(Y_train,color="b")
plt.show()
