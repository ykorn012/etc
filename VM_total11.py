# pls Linear Regression Model
# pls update 하고 VM ez = y - y_hat 출력 (lamda_PLS = 0.1)
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale

#os.chdir("D:/01. CLASS/Machine Learning/01. FabWideSimulation3/")
pls = PLSRegression(n_components=6, scale=True, max_iter=1000)
lamda_PLS = 0.1
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

N = 120
DoE_Queue = []
Drift_Queue = []

def sampling_drift(k1_p1, k2_p1):
    k_p1 = np.array([[k1_p1], [k2_p1]])
    return np.sum(k_p1 * d_p1, axis=0)

def sampling(k, isDrift):
    u1_p1 = np.random.normal(0.4, np.sqrt(0.2))
    u2_p1 = np.random.normal(0.6, np.sqrt(0.2))
    u_p1 = np.array([u1_p1, u2_p1])

    v1_p1 = np.random.normal(1, np.sqrt(0.2))
    v2_p1 = 2 * v1_p1
    v3_p1 = np.random.uniform(0.2, 1.2)
    v4_p1 = 3 * v3_p1
    v5_p1 = np.random.uniform(0, 0.4)
    v6_p1 = np.random.normal(-0.6, np.sqrt(0.2))

    v_p1 = np.array([v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1])

    if isDrift == True:
        k1_p1 = k % 100  # n = 100 일 때 #1 entity maintenance event
        k2_p1 = k % 200  # n = 200 일 때 #1 entity maintenance event
    else:
        k1_p1 = k
        k2_p1 = k

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.1))
    e2_p1 = np.random.normal(0, np.sqrt(0.2))
    e_p1 = np.array([e1_p1, e2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + e_p1
    rows = np.r_[psi, y_p1]
    return rows

def pls_update(V, Y):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in cv.split(V):
        V_train = V[train_index]
        Y_train = Y[train_index]
        V_test = V[test_index]
        Y_test = Y[test_index]
        pls.fit(V_train, Y_train[:, 0:2])
    return pls

def plt_show(n, y1_act, y1_pred):
    plt.plot(np.arange(n), y1_act, 'bx--', y1_pred,'rx--', linewidth=2)
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

for k in range(0, N): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k, True)
    DoE_Queue.append(result)

npDoE_Queue = np.array(DoE_Queue)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:8]
Y0 = plsModelData[:,10:12]

X_train, X_test, y_train, y_test = train_test_split(V0, Y0, test_size = 20)

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

for i in range(N):
    k1 = npDoE_Queue[i:i + 1, 8:9]
    k2 = npDoE_Queue[i:i + 1, 9:10]
    drift = sampling_drift(k1.reshape(1)[0], k2.reshape(1)[0])
    Drift_Queue.append(drift)

npDrift_Queue = np.array(Drift_Queue)

y_pred = pls.predict(V0) + DoE_Mean[10:12] + npDrift_Queue
y_act = npDoE_Queue[:,10:12]  + npDrift_Queue

print("Mean squared error: %.3f" % mean_squared_error(y_act, y_pred))
#plt_show(N, y_act[:,0:1], y_pred[:,0:1])

meanV0 = DoE_Mean[0:8]
meanY0 = DoE_Mean[10:12]

plsWindow = DoE_Queue.copy()

meanVz = meanV0
meanYz = meanY0  ## V0, Y0 Mean Center
Z = 40
M = 10
M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []


#### 고려사항
## 1. Size N에 따라 update 해주자...  -m을 빼주고.. 거기에 추가하자..


for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k, True)
        psiK = result[0:8]
        psiKStar = psiK - meanVz

        k1 = result[8:9][0]
        k2 = result[9:10][0]
        drift = sampling_drift(k1, k2)

        result[10:12] = result[10:12] + drift

        y_predK = pls.predict(psiKStar.reshape(1, 8)) + meanYz + drift
#        print("act = ", result[10:12], "meanYz = ", meanYz, "diff = ", result[10:12] - meanYz)

        rows = np.r_[result, y_predK.reshape(2,)]
        M_Queue.append(rows)

        sample = np.array(M_Queue)

        if k % M == 0:
            y1_pred1.append(rows[10:12])
        else:
            y1_pred1.append(rows[12:14])
        y1_act1.append(rows[10:12])


    del plsWindow[0:M]

    ez = M_Queue[M - 1][10:12] - M_Queue[M - 1][12:14]
    ez_Queue.append(ez)

#    print("Z = ", z, "k = ", k, "act = ", M_Queue[M - 1][10:12], "pred = ", M_Queue[M - 1][12:14], "ez1 = ", ez[0])

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:10] = lamda_PLS * npM_Queue[0:M - 1, 0:10]
    npM_Queue[0:M - 1, 10:12] = lamda_PLS * (npM_Queue[0:M - 1, 12:14] + 0.5 * ez)
    npM_Queue.resize(M, 12)

    for i in range(M):
        plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(np.array(plsWindow), axis=0)
    meanVz = M_Mean[0:8]
    meanYz = M_Mean[10:12]
    #print("k : ", k, ", z : ", z, ", meanVz : ", meanVz[8:10], ", meanYz : ", meanYz)

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:8]
    Y = plsModelData[:, 10:12]

    #pls = pls_update(V, Y)
    pls.fit(V, Y)

    del M_Queue[0:M]

y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)
# sample = np.c_[y1_act, y1_pred]
#np.savetxt("output/vm_total.csv", plsWindow, delimiter=",", fmt="%s")

print("Mean squared error: %.3f" % mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))

plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])


ot = np.array(ez_Queue)

#plt.plot(np.arange(Z + 1), ot[:,0:1], 'bs-', ot[:,1:2], 'rs--', linewidth=2)
# plt.plot(np.arange(Z + 1), ot[:,0:1], 'rs--', linewidth=2)
# plt.xlabel('Metrology Run No.(z)')
# plt.ylabel('Ez')
