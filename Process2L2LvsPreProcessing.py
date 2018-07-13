import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
init_lamda_PLS = 1
lamda_PLS = 1

Tgt = np.array([0, 0])
A_p1 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p1 = np.array([[0, 0.05], [0, 0.05]])
C_p1 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p1 = np.array([[0.05, 0], [0, 0.05]])

sample_init_EP = []
sample_vm_EP = []
sample_init_VP = []
sample_vm_VP = []

preProcessMet = np.genfromtxt('D:/11. Programming/ML/process1-metrology.csv', delimiter=',')

np.random.seed(5000000)

I = np.identity(2)

L1_SC = 0.45
L2_SC = 0.35

IL1 = L1_SC * I
IL2 = L2_SC * I

AL1 = L1_SC * I
AL2 = L2_SC * I

Z = 10
M = 10
N = Z * M

def sampling_vp():
    v1_p1 = np.random.normal(-0.4, np.sqrt(0.2))
    v2_p1 = 2 * v1_p1
    v3_p1 = np.random.uniform(0.2, 0.6)
    v4_p1 = 3 * v3_p1
    v5_p1 = np.random.uniform(0, 0.4)

    v_p1 = np.array([v1_p1, v2_p1, v3_p1, v4_p1, v5_p1])
    return v_p1

def sampling_ep():
    e1_p1 = np.random.normal(0, np.sqrt(0.05))
    e2_p1 = np.random.normal(0, np.sqrt(0.1))
    e_p1 = np.array([e1_p1, e2_p1])
    return e_p1

def sampling(k, uk = np.array([0, 0]), vp = np.array([0, 0, 0, 0, 0]), ep = np.array([0, 0]), initialVM = True):
    u1_p1 = uk[0]
    u2_p1 = uk[1]
    u_p1 = uk

    v1_p1 = vp[0]
    v2_p1 = vp[1]
    v3_p1 = vp[2]
    v4_p1 = vp[3]
    v5_p1 = vp[4]

    v_p1 = vp
    e_p1 = ep

    if initialVM == True:
        k1_p1 = k
        k2_p1 = k
    else:
        k1_p1 = k
        k2_p1 = k

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, k1_p1, k2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    rows = np.r_[psi, y_p1]

    return rows

def sampling2(k, uk = np.array([0, 0]), vp = np.array([0, 0, 0, 0, 0]), ep = np.array([0, 0]), initialVM = True):
    u1_p1 = uk[0]
    u2_p1 = uk[1]
    u_p1 = uk

    v1_p1 = vp[0]
    v2_p1 = vp[1]
    v3_p1 = vp[2]
    v4_p1 = vp[3]
    v5_p1 = vp[4]

    v_p1 = vp
    e_p1 = ep

    if initialVM == True:
        k1_p1 = k
        k2_p1 = k
    else:
        k1_p1 = k
        k2_p1 = k

    k_p1 = np.array([[k1_p1], [k2_p1]])

    f1_p1 = preProcessMet[k - 1][0]
    f2_p1 = preProcessMet[k - 1][1]

    # f1_p1 = 200
    # f2_p1 = 100

    f_p1 = np.array([f1_p1, f2_p1])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, k1_p1, k2_p1, f1_p1, f2_p1])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1 + f_p1.dot(F_p1)

    # print("f_p1 : ", f_p1)
    # print("f_p1.dot(F_p1) : ", f_p1.dot(F_p1))
    # print("y_p1 : ", y_p1)

    rows = np.r_[psi, y_p1]

    return rows

def pls_update(V, Y):
    pls.fit(V, Y)
    return pls

def plt_show1(n, y1_act):
    plt.figure()
    plt.plot(np.arange(1, n + 1), y1_act, 'bx--', lw=2, ms=10, mew=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

def plt_show2(n, y1_act):
    plt.plot(np.arange(1, n + 1), y1_act, 'ro-', linewidth=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

for k in range(0, N + 1):
    sample_init_VP.append(sampling_vp())
    sample_init_EP.append(sampling_ep())

DoE_Queue = []
uk_next = np.array([0, 0])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

vp_next = sample_init_VP[0]
ep_next = sample_init_EP[0]

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(k, uk_next, vp_next, ep_next, True)
    npResult = np.array(result)

    L1 = IL1
    L2 = IL2

    # ================================== R2R Control =====================================
    uk = npResult[0:2]
    yk = npResult[9:11]

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)
#    result[10:12] = uk.dot(A_p1) + Dk + Kd

    Kd_prev = Kd
    Dk_prev = Dk

    if k % M == 0:
        uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
        vp_next = sample_init_VP[k]
    ep_next = sample_init_EP[k]
#    print("uk_next :", uk_next)
    DoE_Queue.append(result)

initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)

plsWindow = []

for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:9] = init_lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:9]
    npPlsWindow[z * M:(z + 1) * M - 1, 9:11] = init_lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 9:11])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:9]
Y0 = plsModelData[:,9:11]

pls = pls_update(V0, Y0)

print('Init L2L-Actual Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[9:11]
y_act = npDoE_Queue[:,9:11]

print("Init L2L-Actual Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("Init L2L-Actual r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

#plt_show1(Z * M, y_act[:,1:2])

#==================================== VM + R2R =======================================
Z = 20
M = 10
N = Z * M
for k in range(0, N + 1):
    sample_vm_VP.append(sampling_vp())
    sample_vm_EP.append(sampling_ep())

meanVz = DoE_Mean[0:9]
meanYz = DoE_Mean[9:11]  ## V0, Y0 Mean Center

yk = np.array([0, 0])

Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

Dk = np.array([0, 0])
Kd = np.array([0, 0])

uk_next = np.array([0, 0])
vp_next = sample_vm_VP[0]
ep_next = sample_vm_EP[0]

M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []


for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling(k, uk_next, vp_next, ep_next, False)
        psiK = result[0:9]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 9)) + meanYz
        rows = np.r_[result, y_predK.reshape(2,)]

        y1_pred1.append(rows[11:13])
        y1_act1.append(rows[9:11])

        # ================================== VM + R2R =====================================
        L1 = AL1
        L2 = AL2

        if k % M != 0:
            yk = rows[11:13]
        else:
            yk = rows[9:11]
        uk = psiK[0:2]

        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

#        rows[10:12] = uk.dot(A_p1) + Dk + Kd
#        y1_act1.append(yk)

        Kd_prev = Kd
        Dk_prev = Dk

        ep_next = sample_vm_EP[k]

        M_Queue.append(rows)

    del plsWindow[0:M]

    uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
    uk_next = uk_next.reshape(2, )
    vp_next = sample_vm_VP[k]

    if z == 0:
        ez = 0

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:9] = lamda_PLS * npM_Queue[0:M - 1, 0:9]
    npM_Queue[0:M - 1, 9:11] = lamda_PLS * (npM_Queue[0:M - 1, 11:13] + 0.5 * ez)
    npM_Queue = npM_Queue[:, 0:11]

    for i in range(M):
       plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(plsWindow, axis=0)
    meanVz = M_Mean[0:9]
    meanYz = M_Mean[9:11]

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:9]
    Y = plsModelData[:, 9:11]

    pls_update(V, Y)

    ez = M_Queue[M - 1][9:11] - M_Queue[M - 1][11:13]
    ez_Queue.append(ez)

    del M_Queue[0:M]

y1_act = np.array(y1_act1)
y1_pred = np.array(y1_pred1)

print("L2L-Actual Mean squared error: %.3f" % metrics.mean_squared_error(y1_act[:,1:2], y1_pred[:,1:2]))
print("L2L-Actual r2 score: %.3f" % metrics.r2_score(y1_act[:,1:2], y1_pred[:,1:2]))

#plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])
plt_show1(Z * M, y1_act[:,1:2])

###============================= R2R-VM With PreProcessing =================================
###============================= R2R-VM With PreProcessing =================================
###============================= R2R-VM With PreProcessing =================================

pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)

Z = 10
M = 10
N = Z * M

DoE_Queue = []
# uk_next = np.array([-50, -100])
# Dk_prev = np.array([20, 25])
uk_next = np.array([0, 0])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

sample_init_VP = []
sample_init_EP = []

for k in range(0, N + 1):
    sample_init_VP.append(sampling_vp())
    sample_init_EP.append(sampling_ep())
vp_next = sample_init_VP[0]
ep_next = sample_init_EP[0]

for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling2(k, uk_next, vp_next, ep_next, True)
    npResult = np.array(result)

    L1 = IL1
    L2 = IL2

    # ================================== R2R Control With PreProcessing =====================================
    uk = npResult[0:2]
    yk = npResult[11:13]

    Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)
    Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)
#    result[10:12] = uk.dot(A_p1) + Dk + Kd

    Kd_prev = Kd
    Dk_prev = Dk

    uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
    # print("uk_next :", uk_next)
    # print("Kd_prev :", Kd_prev)
    # print("Dk_prev :", Dk_prev)
    vp_next = sample_init_VP[k]
    ep_next = sample_init_EP[k]

    DoE_Queue.append(result)

initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)

plsWindow = []

for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:11] = init_lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:11]
    npPlsWindow[z * M:(z + 1) * M - 1, 11:13] = init_lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 11:13])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:11]
Y0 = plsModelData[:,11:13]

pls = pls_update(V0, Y0)

print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[11:13]
y_act = npDoE_Queue[:,11:13]

print("Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

#plt_show2(Z * M, y_act[:,1:2])

#==================================== VM + R2R With PreProcessing=======================================

Z = 20
M = 10
N = Z * M

meanVz = DoE_Mean[0:11]
meanYz = DoE_Mean[11:13]  ## V0, Y0 Mean Center

yk = np.array([0, 0])

# uk_next = np.array([-50, -100])
# Dk_prev = np.array([20, 25])

uk_next = np.array([0, 0])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])

Dk = np.array([0, 0])
Kd = np.array([0, 0])

M_Queue = []
ez_Queue = []
ez_Queue.append([0,0])  #e0 = (0,0)
y1_act1 = []
y1_pred1 = []

sample_vm_VP = []
sample_vm_EP = []

for k in range(0, N + 1):
    sample_vm_VP.append(sampling_vp())
    sample_vm_EP.append(sampling_ep())
vp_next = sample_vm_VP[0]
ep_next = sample_vm_EP[0]

for z in np.arange(0, Z):
    for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
        result = sampling2(k, uk_next, vp_next, ep_next, False)
        psiK = result[0:11]
        psiKStar = psiK - meanVz
        y_predK = pls.predict(psiKStar.reshape(1, 11)) + meanYz
        rows = np.r_[result, y_predK.reshape(2,)]

        y1_pred1.append(rows[13:15])
        y1_act1.append(rows[11:13])

        # ================================== R2R Control With PreProcessing =====================================
        L1 = AL1
        L2 = AL2

        if k % M != 0:
            yk = rows[13:15]
        else:
            yk = rows[11:13]
        uk = psiK[0:2]

        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot((I - L1))
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)

#        rows[11:13] = uk.dot(A_p1) + Dk + Kd
#        y1_act1.append(yk)

        Kd_prev = Kd
        Dk_prev = Dk

        uk_next = (Tgt - Dk - Kd).dot(np.linalg.inv(A_p1))
        uk_next = uk_next.reshape(2, )
        vp_next = sample_vm_VP[k]
        ep_next = sample_vm_EP[k]

        M_Queue.append(rows)

    del plsWindow[0:M]

    if z == 0:
        ez = 0

    npM_Queue = np.array(M_Queue)
    npM_Queue[0:M - 1, 0:11] = lamda_PLS * npM_Queue[0:M - 1, 0:11]
    npM_Queue[0:M - 1, 11:13] = lamda_PLS * (npM_Queue[0:M - 1, 13:15] + 0.5 * ez)
    npM_Queue = npM_Queue[:, 0:13]

    for i in range(M):
       plsWindow.append(npM_Queue[i])

    M_Mean = np.mean(plsWindow, axis=0)
    meanVz = M_Mean[0:11]
    meanYz = M_Mean[11:13]

    plsModelData = plsWindow - M_Mean
    V = plsModelData[:, 0:11]
    Y = plsModelData[:, 11:13]

    pls_update(V, Y)

    ez = M_Queue[M - 1][11:13] - M_Queue[M - 1][13:15]
    ez_Queue.append(ez)

    del M_Queue[0:M]

y1_act = np.array(y1_act1)

y1_pred = np.array(y1_pred1)

print("Mean squared error: %.3f" % metrics.mean_squared_error(y1_act[:,1:2], y1_pred[:,1:2]))
print("r2 score: %.3f" % metrics.r2_score(y1_act[:,1:2], y1_pred[:,1:2]))

#plt_show(Z * M, y1_act[:,0:1], y1_pred[:,0:1])
plt_show2(Z * M, y1_act[:,1:2])
