import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics


os.chdir("D:/11. Programming/ML/01. FabWideSimulation8/")
pls = PLSRegression(n_components=6, scale=False, max_iter=500000, copy=True)
lamda_PLS = 1
Tgt = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

I = np.identity(2)

# L1 = 0.55 * I
# L2 = 0.75 * I

L1 = 1 * I
L2 = 0 * I


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
        k1_p1 = k
        k2_p1 = k

    k_p1 = np.array([[k1_p1], [k2_p1]])

    psi = np.array([u1_p1, u2_p1, v1_p1, v2_p1, v3_p1, v4_p1, v5_p1, v6_p1, k1_p1, k2_p1])

    e1_p1 = np.random.normal(0, np.sqrt(0.2))
    e2_p1 = np.random.normal(0, np.sqrt(0.4))

    if initialVM:
        e_p1 = np.array([0, 0])
#        e_p1 = np.array([e1_p1, e2_p1])
    else:
#        e_p1 = np.array([e1_p1, e2_p1])
        e_p1 = np.array([0, 0])

    y_p1 = u_p1.dot(A_p1) + v_p1.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + e_p1
    rows = np.r_[psi, y_p1]

    print('k : ', k)
    print('yk : %.5f' % y_p1[0])
    print('u_p1 : ', u_p1)
    print('u_p1.dot(A_p1) : %.5f' % u_p1.dot(A_p1)[0])
    print('v_p1 : ', v_p1)
    print('v_p1.dot(C_p1) : %.5f' % v_p1.dot(C_p1)[0])
    print('np.sum(k_p1 * d_p1, axis=0) : %.5f' % np.sum(k_p1 * d_p1, axis=0)[0])

    return rows

def pls_update(V, Y):
    pls.fit(V, Y)
    return pls

def plt_show(n, y1_act, y1_pred):
    plt.plot(np.arange(n), y1_act, 'bx--', y1_pred,'rx--', linewidth=2)
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

def plt_show1(n, y1_act):
    plt.plot(np.arange(1, n + 1), y1_act, 'ro-', linewidth=2)
    plt.xticks(np.arange(0, n, 10))
    plt.xlabel('Run No.')
    plt.ylabel('y_value')

Z = 2
M = 10
N = Z * M
DoE_Queue = []
uk_next = np.array([0, 0])
Dk_prev = np.array([0, 0])
Kd_prev = np.array([0, 0])
vk_next = np.array([0, 0, 0, 0, 0, 0])

vk_next = sampling_vp()

index = 1
for k in range(1, N + 1): # range(101) = [0, 1, 2, ..., 100])
    result = sampling(index, uk_next, vk_next, True)
    # print('result_k : ', result[10:12])
    # print('uk_next : ', uk_next)
    npResult = np.array(result)
    DoE_Queue.append(result)

    # vk_next = sampling_vp()

    # if k % M == 0:
    #     yk = npResult[10:12]
    #     print('yk : ', yk)
    #     uk = npResult[0:2]
    #     vk = npResult[2:8]
    #     k_p1 = npResult[8:10]
    #     temp = vk.dot(C_p1) + np.sum(k_p1 * d_p1, axis=0) + 0.15
    #     uk_next = (Tgt - temp).dot(np.linalg.inv(A_p1))


    #================================== R2R Control =====================================
    if k % M == 0:
        yk = npResult[10:12]
        uk = npResult[0:2]
        print('uk.dot(A_p1) : %.5f' % uk.dot(A_p1)[0])
        print('yk : %.5f' % yk[0])
        Dk = (yk - uk.dot(A_p1)).dot(L1) + Dk_prev.dot(I - L1)  #v(k)C + kd
        Kd = (yk - uk.dot(A_p1) - Dk_prev).dot(L2) + Kd_prev.dot(I - L2)
        #Kd = 0.15

        uk_next = (Tgt - (Dk + Kd)).dot(np.linalg.inv(A_p1))
        vk_next = sampling_vp()
        index = index + M

        print('Dk : ', Dk)
        print('Kd : ', Kd)

        Kd_prev = Kd
        Dk_prev = Dk



initplsWindow = DoE_Queue.copy()
npPlsWindow= np.array(initplsWindow)
plsWindow = []


for z in np.arange(0, Z):
    npPlsWindow[z * M:(z + 1) * M - 1, 0:10] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:10]
    npPlsWindow[z * M:(z + 1) * M - 1,10:12] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 10:12])

for i in range(len(npPlsWindow)):
    plsWindow.append(npPlsWindow[i])

npDoE_Queue = np.array(plsWindow)
DoE_Mean = np.mean(npDoE_Queue, axis = 0)

plsModelData = npDoE_Queue - DoE_Mean
V0 = plsModelData[:,0:10]
Y0 = plsModelData[:,10:12]

pls = pls_update(V0, Y0)
#np.savetxt("output/npDoE_Queue1.csv", npDoE_Queue, delimiter=",", fmt="%s")
print('Coefficients: \n', pls.coef_)

y_pred = pls.predict(V0) + DoE_Mean[10:12]
y_act = npDoE_Queue[:,10:12]

print("Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
print("r2 score: %.3f" % metrics.r2_score(y_act, y_pred))
#plt_show(N, y_act[:,0:1], y_pred[:,0:1])
plt_show1(Z * M, y_act[:,0:1])
#
# np1PlsWindow= np.array(initplsWindow)
# V0 = np1PlsWindow[:,0:10]
# y_predK = pls.predict(V0)
# plt_show1(Z * M, y_predK[:,0:1])


