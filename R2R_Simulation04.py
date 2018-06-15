import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics


os.chdir("D:/11. Programming/ML/01. FabWideSimulation7/")
pls = PLSRegression(n_components=6, scale=False, max_iter=1500, copy=True)
lamda_PLS = 0.1
Tgt = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

# L1 = 0.5 * np.identity(2)
# L2 = 0.5 * np.identity(2)
I = np.identity(2)
L1 = 0.55 * I
L2 = 0.75 * I

def sampling_up():
    u1_p1 = np.random.normal(0.4, np.sqrt(0.2))
    u2_p1 = np.random.normal(0.6, np.sqrt(0.2))
    u_p1 = np.array([u1_p1, u2_p1])
    return u_p1

def sampling_vp():
    v1_p1 = np.random.normal(1, np.sqrt(0.2))
    v2_p1 = 2 * v1_p1
    v3_p1 = np.random.uniform(0.2, 1.2)
    v4_p1 = 3 * v3_p1
    v5_p1 = np.random.uniform(0, 0.4)
    v6_p1 = np.random.normal(-0.6, np.sqrt(0.2))

    v_p1 = np.array([v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1])
    return v_p1

def sampling(k, uk = np.array([0, 0]), vp = np.array([0, 0, 0, 0, 0, 0]), initialVM = True):
    u1_p1 = uk[0]
    u2_p1 = uk[1]
    u_p1 = uk

    v1_p1 = vp[0]
    v2_p1 = vp[1]
    v3_p1 = vp[2]
    v4_p1 = vp[3]
    v5_p1 = vp[4]
    v6_p1 = vp[5]

    v_p1 = vp

    if initialVM == True:
        k1_p1 = k
        k2_p1 = k
    else:
        k1_p1 = k  # n = 100 일 때 #1 entity maintenance event
        k2_p1 = k  # n = 200 일 때 #1 entity maintenance event

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.2))
    e2_p1 = np.random.normal(0, np.sqrt(0.4))

    if initialVM:
        e_p1 = np.array([0, 0])
#        e_p1 = np.array([e1_p1, e2_p1])
    else:
        e_p1 = np.array([e1_p1, e2_p1])
#        e_p1 = np.array([0, 0])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    # if y_p1[0] > 1 or y_p1[0] < -1:
    #     print('k : ', k)
        # print('yk : %.5f' % y_p1[0])
        # print('u_p1.dot(A_p1) : %.5f' % u_p1.dot(A_p1)[0])
        # print('v_p1.dot(C_p1) : %.5f' % v_p1.dot(C_p1)[0])
        # print('np.sum(k_p1 * d_p1, axis=0) : %.5f' % np.sum(k_p1 * d_p1, axis=0)[0])
    rows = np.r_[psi, y_p1]

    return rows

def pls_update(V, Y):
    pls.fit(V, Y)
    return pls

def plt_show(n, y1_act, y1_pred):
    plt.plot(np.arange(1, n + 1), y1_act, 'bx--', y1_pred,'ro--', linewidth=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

def plt_show1(n, y1_act):
    plt.plot(np.arange(1, n + 1), y1_act, 'ro-', linewidth=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

N = 100
DoE_Queue = []
uk_next = np.array([0, 0])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k, uk_next, sampling_vp(), True)
#    print('yk : %.5f' % result[10:11])
    while (result[10:11] > 1) or (result[10:11] < -1):
        result = sampling(k, uk_next, sampling_vp(), True)
    DoE_Queue.append(result)


    # ================================== R2R Control =====================================
    npResult = np.array(result)

    yk = npResult[10:12]
    uk = npResult[0:2]

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

    uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
#    temp = sampling(k, uk_next, npResult[2:8], True)
#    print('yhatk : %.5f' % temp[10:11])

    Kd_prev = Kd
    Dk_prev = Dk

initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)

plsWindow = []
Z = 10
M = 10

#np.savetxt("output/vm_sample1.csv", DoE_Queue, delimiter=",", fmt="%s")

for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:10] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:10]
    npPlsWindow[z * M:(z + 1) * M - 1,10:12] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 10:12])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)
#np.savetxt("output/vm_sample2.csv", plsWindow, delimiter=",", fmt="%s")
plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:12]

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[10:12]
y_act = npDoE_Queue[:,10:12]

print("Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("r2 score: %.3f" % metrics.r2_score(y_act, y_pred))
#plt_show(N, y_act[:,0:1], y_pred[:,0:1])
#plt_show1(N, y_act[:,0:1])

meanVz = DoE_Mean[0:10]
meanYz = DoE_Mean[10:12]  ## V0, Y0 Mean Center

yk = np.array([0, 0])

Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

Dk = np.array([0, 0])
Kd = np.array([0, 0])
uk_next = np.array([0, 0])
Z = 10
M = 10
M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []

for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k, uk_next, sampling_vp(), False)
        # while (result[10:11] > 1) or (result[10:11] < -1):
        #     result = sampling(k, uk_next, sampling_vp(), False)
        psiK = result[0:10]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 10)) + meanYz
#        print("k : ", k, ", z : ", z, ", psiKStar : ", psiKStar[8:10], ", y_predK : ", y_predK)
        rows = np.r_[result, y_predK.reshape(2,)]
        M_Queue.append(rows)

        # if k % M == 0:
        #     y1_pred1.append(rows[10:12])
        # else:
        #     y1_pred1.append(rows[12:14])
        y1_pred1.append(rows[12:14])
        y1_act1.append(rows[10:12])

        # ================================== R2R Control =====================================
        yk = y_predK
        uk = psiK[0:2]

        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

        uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
        uk_next = uk_next.reshape(2,)
        print("uk_next : ", uk_next)
        print("uk_A_p1 : ", uk_next.dot(A_p1))

        Kd_prev = Kd
        Dk_prev = Dk

    del plsWindow[0:M]

    ez = M_Queue[M - 1][10:12] - M_Queue[M - 1][12:14]
    ez_Queue.append(ez)

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:10] = lamda_PLS * npM_Queue[0:M - 1, 0:10]
    npM_Queue[0:M - 1, 10:12] = lamda_PLS * (npM_Queue[0:M - 1, 12:14] + 0.5 * ez)
    npM_Queue = npM_Queue[:, 0:12]

    for i in range(M):
       plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(plsWindow, axis=0)
    meanVz = M_Mean[0:10]
    meanYz = M_Mean[10:12]

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:10]
    Y = plsModelData[:, 10:12]

#    pls = copy.deepcopy(pls_udt)
    pls_update(V, Y)
#    print(pls.coef_)

    del M_Queue[0:M]

#np.savetxt("output/vm_sample.csv", plsWindow, delimiter=",", fmt="%s")
y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)


print("Mean squared error: %.3f" % metrics.mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))
print("r2 score: %.3f" % metrics.r2_score(y1_act[:,0:1], y1_pred[:,0:1]))

#plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])
plt_show1(Z * M, y1_act[:,0:1])

met_run = np.array(ez_Queue)

# plt.plot(np.arange(Z + 1), met_run[:,0:1], 'bs-', met_run[:,1:2], 'rs--', linewidth=2)
# #plt.plot(np.arange(Z + 1), met_run[:,0:1], 'bs-', linewidth=2)
# plt.xlabel('Metrology Run No.(z)')
# plt.ylabel('Ez')
