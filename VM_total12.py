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
pls = PLSRegression(n_components=6, scale=True, max_iter=500)
lamda_PLS = 1
Tgt = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

L1 = 0.55 * np.identity(2)
L2 = 0.75 * np.identity(2)
I = np.identity(2)

N = 100
DoE_Queue = []

def sampling(k, isR2R, uk = np.array([0, 0])):
    if isR2R:
        u1_p1 = uk[0]
        u2_p1 = uk[1]
        u_p1 = uk
    else:
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

    k1_p1 = 1
    k2_p1 = 1

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.2))
    e2_p1 = np.random.normal(0, np.sqrt(0.4))
    e_p1 = np.array([e1_p1, e2_p1])
#    e_p1 = [0,0]

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    rows = np.r_[psi, y_p1]
    return rows

def pls_update(V, Y):
    cv = KFold(n_splits=50, shuffle=True, random_state=42)

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

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k, False)
    DoE_Queue.append(result)

npDoE_Queue = np.array(DoE_Queue)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:12]

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[10:12]
y_act = npDoE_Queue[:,10:12]

print("Mean squared error: %.3f" % mean_squared_error(y_act, y_pred))
#plt_show(N, y_pred[:,0:1], y_pred[:,0:1])

meanV0 = DoE_Mean[0:10]
meanY0 = DoE_Mean[10:12]

plsWindow = DoE_Queue.copy()

meanVz = meanV0
meanYz = meanY0  ## V0, Y0 Mean Center

yk = meanY0
uk = DoE_Mean[0:2]

Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])
uk_next = meanV0[0:2].dot(A_p1)
uk_next = (Tgt - yk - uk.dot(A_p1)).dot(np.linalg.inv(A_p1))

Z = 20
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
        result = sampling(k, True, uk_next)
        psiK = result[0:10]
#        print("psiK : ", psiK[0:2])
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 10)) + meanYz
        #print("act = ", result[10:12], "meanYz = ", meanYz, "diff = ", result[10:12] - meanYz)

        rows = np.r_[result, y_predK.reshape(2,)]
        M_Queue.append(rows)

        sample = np.array(M_Queue)

        y1_pred1.append(rows[12:14])
        y1_act1.append(rows[10:12])


#    del plsWindow[0:M]

    npM_Queue = np.array(M_Queue)

    V = npM_Queue[:, 0:10]
    Y = npM_Queue[:, 10:12]

    #pls = pls_update(V, Y)
    pls.fit(V, Y)

    #================================== R2R Control =====================================
    yk = npM_Queue[M - 1][10:12]
    uk = npM_Queue[M - 1][0:2]

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

    uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
    print("uk_next : ", uk_next)
    print("uk_A_p1 : ", uk_next.dot(A_p1))

    Kd_prev = Kd
    Dk_prev = Dk

    del M_Queue[0:M]


y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)
# sample = np.c_[y1_act, y1_pred]
#np.savetxt("output/vm_total.csv", plsWindow, delimiter=",", fmt="%s")

print("Mean squared error: %.3f" % mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))

plt_show(Z * M, y1_act[:,0:1], y1_act[:,0:1])

