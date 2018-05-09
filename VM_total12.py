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

lamda_PLS = 0.1
N = 120
DoE_Queue = []

def sampling(k):
    A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
    d_p1 = np.array([[0.1, 0], [0.05, 0]])
    C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

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

    k1_p1 = k % 100  # n = 100 일 때 #1 entity maintenance event
    k2_p1 = k % 200  # n = 200 일 때 #1 entity maintenance event

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.1))
    e2_p1 = np.random.normal(0, np.sqrt(0.2))
    e_p1 = np.array([e1_p1, e2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    rows = np.r_[psi, y_p1]
    return rows

def plt_show(y1_act, y1_pred):
    plt.plot(np.arange(N), y1_act, 'bx--', y1_pred,'rx--', linewidth=2)
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
# plt_show(y1_act, y1_pred)

actV0 = npDoE_Queue[:,0:10]
actY0 = npDoE_Queue[:,10:12]
predY0 = np.array(Y_sort[:,3:5] + DoE_Mean[10:12])
meanV0 = DoE_Mean[0:10]
meanY0 = DoE_Mean[10:12]


def pls_Update(queue, ez, mean_Z):
    np_queue = np.array(queue)
    np_queue[1:10, 0:10] = lamda_PLS * np_queue[1:10, 0:10]
    np_queue[1:10, 12:14] = lamda_PLS * (np_queue[1:10, 12:14] + 0.5 * ez)
    up_queue = np_queue - mean_Z
    V = up_queue[:, 0:10]
    y = up_queue[:, 12:14]
    pls.fit(V, y)


W_Queue = []
meanV = meanV0
meanY = meanY0  ## V0, Y0 Mean Center
Z = 1
M = 10
ez = []
ez.append([0,0])
result_k = []

ez_Queue = []
y1_act1 = []
y1_pred1 = []
mean_Z = np.zeros((14,))

N = 400

#### 고려사항
## 1. Size N에 따라 update 해주자...  -m을 빼주고.. 거기에 추가하자..


for z in np.arange(0, Z):
    if z > 0:
        mean_Z = np.mean(W_Queue, axis=0)  # new Vz, Yz
        y_zm_hat = W_Queue[z * M - 1][12:14] + mean_Z[12:14]

        y_zm = W_Queue[z * M - 1][10:12]
        ez_v = y_zm - y_zm_hat

        ez.append(ez_v)

        pls_Update(W_Queue, ez_v, mean_Z)

#       del W_Queue[1:M + 1]

    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        resultV = sampling(k)
        Vz = resultV[0:10] - meanV

        y_pred = pls.predict(Vz.reshape(1, 10)) + meanY

        # rows = np.r_[result_k, y_pred.reshape(2,)]
        # #print(rows)
        # y1_pred1.append(rows[12:14])
        # y1_act1.append(rows[10:12])
        # W_Queue.append(rows)

        print(y_pred)

#y_value = np.array(ez[0])

# y1_act = np.array(y1_act1)
# y1_pred = np.array(y1_pred1)
#
# #print("Mean squared error: %.3f" % mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))
#
# plt.plot(np.arange(N), y1_act[:,0:1], 'bx--', y1_pred[:,0:1],'rx--', linewidth=2)
# plt.xlabel('Run No.')
# plt.ylabel('Y_value')
#
# ot = np.array(ez)
#
# plt.plot(np.arange(Z), ot[:,0:1], 'bs-', ot[:,1:2], 'rs--', linewidth=2)
# plt.xlabel('Metrology Run No.(z)')
# plt.ylabel('Ez')
