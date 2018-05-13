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

os.chdir("D:/01. CLASS/Machine Learning/01. FabWideSimulation3/")

lamda_PLS = 0.1
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

N = 120
DoE_Queue = []

def sampling(k):
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
    k1_p1 = (k % 100) * d_p1[0] # n = 100 일 때 #1 entity maintenance event
    k2_p1 = (k % 200) * d_p1[1] # n = 200 일 때 #1 entity maintenance event

    k_p1 = k1_p1 + k2_p1
    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k_p1[0], k_p1[1]])

    e1_p1 = np.random.normal(0, np.sqrt(0.1))
    e2_p1 = np.random.normal(0, np.sqrt(0.2))
    e_p1 = np.array([e1_p1, e2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + k_p1 + e_p1
    rows = np.r_[psi, y_p1]
    return rows

def plt_show(n, y1_act, y1_pred):
    plt.plot(np.arange(n), y1_act, 'bx--', y1_pred,'rx--', linewidth=2)
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

for k in range(0, N): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k)
    DoE_Queue.append(result)

npDoE_Queue = np.array(DoE_Queue)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = np.c_[npDoE_Queue - DoE_Mean, np.arange(N)]

V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:13]

V_train, V_test, Y_train, Y_test = train_test_split(V0, Y0, test_size=0)

pls = PLSRegression(n_components = 6, scale=False)
pls.fit(V_train, Y_train[:,0:2])
Y_pred = pls.predict(V_train)

Y_temp = np.c_[Y_train, Y_pred]
Y_sort = Y_temp[Y_temp[:,2].argsort()]

y1_act = Y_sort[:,:1].reshape(len(Y_train)) + DoE_Mean[10:11]
y1_pred = Y_sort[:,3:4].reshape(len(Y_pred)) + DoE_Mean[10:11]

print("Mean squared error: %.3f" % mean_squared_error(y1_act, y1_pred))
#plt_show(N, y1_act, y1_pred)

predY0 = np.array(Y_sort[:,3:5] + DoE_Mean[10:12])
meanV0 = DoE_Mean[0:10]
meanY0 = DoE_Mean[10:12]


# def pls_Update(m_queue, M):
#
#
#
#
#     np_queue = np.array(queue)
#     np_queue[1:10, 0:10] = lamda_PLS * np_queue[1:10, 0:10]
#     np_queue[1:10, 12:14] = lamda_PLS * (np_queue[1:10, 12:14] + 0.5 * ez)
#     up_queue = np_queue - mean_Z
#     V = up_queue[:, 0:10]
#     y = up_queue[:, 12:14]
#     pls.fit(V, y)

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
    if z > 0:
        del plsWindow[0:M]

        ez = M_Queue[M - 1][10:12] - M_Queue[M - 1][12:14]
        ez_Queue.append(ez)

        npM_Queue = np.array(M_Queue)
        npM_Queue[0:M - 1, 0:10] = lamda_PLS * npM_Queue[0:M - 1, 0:10]
        npM_Queue[0:M - 1, 10:12] = lamda_PLS * (npM_Queue[0:M - 1, 12:14] + 0.5 * ez)

        npM_Queue.resize(M, 12)

        for i in range(M):
            plsWindow.append(npM_Queue[i])

        M_Mean = np.mean(np.array(plsWindow), axis=0)
        #M_Mean[8:10] = [0, 0]
        meanVz = M_Mean[0:10]
        meanYz = M_Mean[10:12]
        print("k : ", k, ", z : ", z, ", meanVz : ", meanVz[8:10], ", meanYz : ", meanYz)

        plsModelData = plsWindow - M_Mean
        V = plsModelData[:, 0:10]
        Y = plsModelData[:, 10:12]

        pls.fit(V, Y)

        # pls_Update(M_Queue, M)
        del M_Queue[0:M]


    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k)
        #print("result : ", result)
        psiK = result[0:10]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 10)) + meanYz

        rows = np.r_[result, y_predK.reshape(2,)]
        M_Queue.append(rows)

        sample = np.array(M_Queue)

        y1_pred1.append(rows[12:14])
        y1_act1.append(rows[10:12])

#y_value = np.array(ez[0])

y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)
# sample = np.c_[y1_act, y1_pred]
np.savetxt("output/vm_total.csv", plsWindow, delimiter=",", fmt="%s")

print("Mean squared error: %.3f" % mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))

plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])

#
# ot = np.array(ez)
#
# plt.plot(np.arange(Z), ot[:,0:1], 'bs-', ot[:,1:2], 'rs--', linewidth=2)
# plt.xlabel('Metrology Run No.(z)')
# plt.ylabel('Ez')
