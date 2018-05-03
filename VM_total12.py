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

#os.chdir("D:/01. CLASS/Machine Learning/01. FabWideSimulation2/")

Y_p1 = []
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))
lamda_PLS = 0.1

N = 120

Queue = []

for k in range(0, N): # range(101) = [0, 1, 2, ..., 100])
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

    k1_p1 = k # n = 100 일 때 #1 entity maintenance event
    k2_p1 = k # n = 200 일 때 #1 entity maintenance event

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.1))
    e2_p1 = np.random.normal(0, np.sqrt(0.2))
    e_p1 = np.array([e1_p1, e2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis = 0) + e_p1
    rows = np.r_[psi, y_p1]
    Queue.append(rows)

Queue = np.array(Queue)
Mean_M = np.mean(Queue, axis = 0)
Sampling_M = Queue - Mean_M

Sampling_M = np.c_[Sampling_M, np.arange(N)]

#np.savetxt("output/vm_total3.csv", Sampling_M, delimiter=",", fmt="%s")

V = Sampling_M[:,0:10]
y = Sampling_M[:,10:13]

T = 7

X_train, X_test, y_train, y_test = train_test_split(V, y, test_size=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# temp = np.c_[X_train, y_train]
#
# np.savetxt("output/vm_total4.csv", temp, delimiter=",", fmt="%s")

kf = KFold(n_splits = 10, shuffle=True, random_state = 1)

# for i in np.arange(1, T):
#     print('====================================================')
#     pls = PLSRegression(n_components = i, scale=False)
#     pls.fit(X_train, y_train[:,0:2])
#     scores = pls.score(X_train, y_train[:,0:2])
#     print('a : ', i, "Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#
#     y_pred = pls.predict(X_train)
#     # The coefficients
#     #print('Coefficients: \n', pls.coef_)
#     ## The mean squared error
#     print("Mean squared error: %.3f" % mean_squared_error(y_train[:,0:2], y_pred))
#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.3f' % r2_score(y_train[:,0:2], y_pred, multioutput='variance_weighted'))
#
# print('====================================================')

pls = PLSRegression(n_components = 6, scale=False)
pls.fit(X_train, y_train[:,0:2])
y_pred = pls.predict(X_train)

y_temp = np.c_[y_train, y_pred]
y_sort = y_temp[y_temp[:,2].argsort()]

#y_pred = pls.transform(X_train, y_train)
#a
# y1_test = y_train_r[:,:1].reshape(len(y_train_r)) + Mean_M[10:11]
# y1_pred = y_pred_r[:,:1].reshape(len(y_pred_r)) + Mean_M[10:11]

y1_test = y_sort[:,:1].reshape(len(y_train)) + Mean_M[10:11]
y1_pred = y_sort[:,3:4].reshape(len(y_pred)) + Mean_M[10:11]

# plt.plot(np.arange(N), y1_test, 'bx--', y1_pred,'rx--', linewidth=2)
# plt.xlabel('Run No.')
# plt.ylabel('Y_value')

def sampling_DOE(k):
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

def pls_Update(queue, ez):
    np_queue = np.array(queue)
    np_queue[1:10, 0:10] = lamda_PLS * np_queue[1:10, 0:10]
    np_queue[1:10, 10:12] = lamda_PLS * (np_queue[1:10, 10:12] + 0.5 * ez)
    Mean_M = np.mean(np_queue, axis=0)
    up_queue = np_queue - Mean_M
    V = up_queue[:, 0:10]
    y = up_queue[:, 10:12]
    pls.fit(V, y)


W_Queue = []
mean_Z = Mean_M  ## V0, Y0 Mean Center
#W_Queue.append(np.array(mean_Z))

#Y_Queue = []

Z = 40
#Z = 3
M = 10
# t = np.array(W_Queue)
# t[1:M,1]
ez = []
ez.append([0,0])
result_k = []

N = 400

#### 고려사항
## 1. 각 예측 값에 0.5 * ez인데.. update 기준도 애매하고.. 10 단위로 넣어줘야 하는지.. n이 헥갈린다.


for z in np.arange(0, Z):
    if z > 0:
        y_zm_hat = W_Queue[z * M - 1][12:14] + mean_Z[10:12]

        #y_p1 = y_zm_hat_t + e_p1  #y_zm_hat에 하는게 이상하다.. 그냥 하는 걸로 고쳐야 한다. Sampling_M(400이랑 같아야 하지 않을까?) 참조

        y_zm = W_Queue[z * M - 1][0:2].dot(A_p1) + W_Queue[9][2:8].dot(C_p1) + np.sum(W_Queue[9][8:10] * d_p1, axis=0) + e_p1
        ez_v = y_zm - y_zm_hat

        ez.append(ez_v)

        pls_Update(W_Queue, ez_v)

        mean_Z = np.mean(W_Queue, axis=0)  # new Vz, Yz
#        del W_Queue[1:M + 1]

    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result_k = sampling_DOE(k - 1)
        Vz = result_k[0:10] - mean_Z.ravel()[0:10]

        y_pred = pls.predict(Vz.reshape(1, 10))
        rows = np.r_[result_k, y_pred.reshape(2,)]
        #print(rows)
        W_Queue.append(rows)
        #print(len(W_Queue))

#y_value = np.array(ez[0])

y1_act = np.array(W_Queue[10:12])
y1_pred = np.array(y1_pred)
#
# plt.plot(np.arange(N), y1_act[:,0:1], 'bx--', y1_pred[:,1:2],'rx--', linewidth=2)
# plt.xlabel('Run No.')
# plt.ylabel('Y_value')

ot = np.array(ez)

plt.plot(np.arange(Z), ot[:,0:1], 'bs-', ot[:,1:2], 'rs--', linewidth=2)
plt.xlabel('Metrology Run No.(z)')
plt.ylabel('Ez')

