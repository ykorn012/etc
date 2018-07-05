import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

os.chdir("D:/11. Programming/ML/")
pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
init_lamda_PLS = 1
lamda_PLS = 1

Tgt = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

np.random.seed(4) # 4

I = np.identity(2)

L1_SC = 0.55
L2_SC = 0.75

IL1 = L1_SC * I
IL2 = L2_SC * I

AL1 = L1_SC * I
AL2 = L2_SC * I

Z = 10
M = 10
N = Z * M

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
        k1_p1 = k
        k2_p1 = k

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.2))
    e2_p1 = np.random.normal(0, np.sqrt(0.4))
    if initialVM:
        e_p1 = np.array([e1_p1, e2_p1])
    else:
        e_p1 = np.array([e1_p1, e2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    rows = np.r_[psi, y_p1]

    return rows

def pls_update(V, Y):
    pls.fit(V, Y)
    return pls

def plt_show1(n, y1_act):
    plt.plot(np.arange(1, n + 1), y1_act, 'bx--', lw=2, ms=8, mew=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

def plt_show2(n, y1_act):
    plt.plot(np.arange(1, n + 1), y1_act, 'ro-', linewidth=1)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')


DoE_Queue = []
uk_next = np.array([-100, 198])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

vp_next = sampling_vp()

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k, uk_next, vp_next, True)
    npResult = np.array(result)

    L1 = IL1
    L2 = IL2

    # ================================== R2R Control =====================================
    uk = npResult[0:2]
    yk = npResult[10:12]

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)
#    result[10:12] = uk.dot(A_p1) + Dk + Kd

    Kd_prev = Kd
    Dk_prev = Dk

    uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
    vp_next = sampling_vp()

    DoE_Queue.append(result)

initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)

plsWindow = []

for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:10] = init_lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:10]
    npPlsWindow[z * M:(z + 1) * M - 1,10:12] = init_lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 10:12])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:12]

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[10:12]
y_act = npDoE_Queue[:,10:12]

print("Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

#plt_show2(Z * M, y_act[:,0:1])

#==================================== VM + R2R =======================================

meanVz = DoE_Mean[0:10]
meanYz = DoE_Mean[10:12]  ## V0, Y0 Mean Center

yk = np.array([0, 0])

Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

Dk = np.array([0, 0])
Kd = np.array([0, 0])
uk_next = np.array([-100, 198])
vp_next = sampling_vp()

Z = 20
M = 10
M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []
y1_act2 = []

for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k, uk_next, vp_next, False)
        psiK = result[0:10]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 10)) + meanYz
        rows = np.r_[result, y_predK.reshape(2,)]

        y1_pred1.append(rows[12:14])

        # ================================== R2R Control =====================================
        L1 = AL1
        L2 = AL2

        if k % M != 0:
            yk = rows[12:14]
        else:
            yk = rows[10:12]
        uk = psiK[0:2]

        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

#        rows[10:12] = uk.dot(A_p1) + Dk + Kd
        y1_act1.append(rows[10:12])

        Kd_prev = Kd
        Dk_prev = Dk

        uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
        uk_next = uk_next.reshape(2, )
        vp_next = sampling_vp()

        M_Queue.append(rows)

    del plsWindow[0:M]

    if z == 0:
        ez = 0

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:10] = lamda_PLS * npM_Queue[0:M - 1, 0:10]
    npM_Queue[0:M - 1, 10:12] = lamda_PLS * (npM_Queue[0:M - 1, 12:14] + 0.5 * ez)
    npM_Queue = npM_Queue[:, 0:12]

    for i in range(M):
        temp = npM_Queue[i:i + 1, 10:12].flatten()
        y1_act2.append(temp)
        plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(plsWindow, axis=0)
    meanVz = M_Mean[0:10]
    meanYz = M_Mean[10:12]

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:10]
    Y = plsModelData[:, 10:12]

    pls_update(V, Y)

    ez = M_Queue[M - 1][10:12] - M_Queue[M - 1][12:14]
    ez_Queue.append(ez)

    del M_Queue[0:M]

y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)

print("Mean squared error: %.3f" % metrics.mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))
print("r2 score: %.3f" % metrics.r2_score(y1_act[:,0:1], y1_pred[:,0:1]))

#plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])
plt_show2(Z * M, y1_act[:,0:1])

np.savetxt("process1-metrology.csv", y1_act2, delimiter=",", fmt="%s")

#==============================  L2L Simulation ==============================

pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)

Z = 10
M = 10
N = Z * M
DoE_Queue = []
uk_next = np.array([-100, 198])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

vp_next = sampling_vp()

ink = 1

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(ink, uk_next, vp_next, True)
    npResult = np.array(result)

    # ================================== L2L Control =====================================
    uk = npResult[0:2]
    yk = npResult[10:12]

    L1 = IL1
    L2 = IL2

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

    Kd_prev = Kd
    Dk_prev = Dk

    if k % M == 0:
        ink = k + 1
        uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
        vp_next = sampling_vp()

    DoE_Queue.append(result)

initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)

plsWindow = []

for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:10] = init_lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:10]
    npPlsWindow[z * M:(z + 1) * M - 1,10:12] = init_lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 10:12])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:12]

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[10:12]
y_act = npDoE_Queue[:,10:12]

print("Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

#plt_show1(Z * M, y_act[:,0:1])

#==================================== VM + L2L =======================================

meanVz = DoE_Mean[0:10]
meanYz = DoE_Mean[10:12]  ## V0, Y0 Mean Center

yk = np.array([0, 0])

Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

Dk = np.array([0, 0])
Kd = np.array([0, 0])
uk_next = np.array([-100, 198])
vp_next = sampling_vp()

Z = 20
M = 10
M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []


for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k, uk_next, vp_next, False)
        psiK = result[0:10]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 10)) + meanYz
        rows = np.r_[result, y_predK.reshape(2,)]

        y1_pred1.append(rows[12:14])

        # ================================== L2L Control =====================================
        L1 = AL1
        L2 = AL2

        if k % M != 0:
            yk = rows[12:14]
        else:
            yk = rows[10:12]
        uk = psiK[0:2]

        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)
        #
        # if k % M != 0:
        #     rows[12:14] = uk.dot(A_p1) + Dk + Kd
        y1_act1.append(rows[10:12])

        Kd_prev = Kd
        Dk_prev = Dk

        if k % M == 0:
            ink = k + 1
            uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
            uk_next = uk_next.reshape(2, )
            vp_next = sampling_vp()

        M_Queue.append(rows)

    del plsWindow[0:M]

    if z == 0:
        ez = 0

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:10] = lamda_PLS * npM_Queue[0:M - 1, 0:10]
    npM_Queue[0:M - 1, 10:12] = lamda_PLS * (npM_Queue[0:M - 1, 12:14])
    npM_Queue = npM_Queue[:, 0:12]

    for i in range(M):
       plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(plsWindow, axis=0)
    meanVz = M_Mean[0:10]
    meanYz = M_Mean[10:12]

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:10]
    Y = plsModelData[:, 10:12]

    pls_update(V, Y)

    ez = M_Queue[M - 1][10:12] - M_Queue[M - 1][12:14]
    ez_Queue.append(ez)

    del M_Queue[0:M]

y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)

print("Mean squared error: %.3f" % metrics.mean_squared_error(y1_act[:,0:1], y1_pred[:,0:1]))
print("r2 score: %.3f" % metrics.r2_score(y1_act[:,0:1], y1_pred[:,0:1]))

#plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])
plt_show1(Z * M, y1_act[:,0:1])




