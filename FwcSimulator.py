import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

class FwcSimulator:

    def __init__(self, Tgt, A, d, C):
        self.pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
        np.random.seed(1000000)
        self.Tgt = Tgt
        self.A = A
        self.d = d
        self.C = C

    def set_VMparameter(self, lamda_PLS):
        self.lamda_PLS = lamda_PLS

    def sampling_vp(self):
        v1 = np.random.normal(1, np.sqrt(0.2))
        v2 = 2 * v1
        v3 = np.random.uniform(0.2, 1.2)
        v4 = 3 * v3
        v5 = np.random.uniform(0, 0.4)
        v6 = np.random.normal(-0.6, np.sqrt(0.2))

        v = np.array([v1, v2, v3, v4, v5, v6])
        return v

    def sampling_ep(self):
        e1 = np.random.normal(0, np.sqrt(0.2))
        e2 = np.random.normal(0, np.sqrt(0.4))
        e = np.array([e1, e2])
        return e

    def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0, 0]), ep=np.array([0, 0]), isInit=True):
        u1 = uk[0]
        u2 = uk[1]
        u = uk

        v1 = vp[0]
        v2 = vp[1]
        v3 = vp[2]
        v4 = vp[3]
        v5 = vp[4]
        v6 = vp[5]

        v = vp
        e = ep

        # if isInit == True:
        #     e_p1 = [0, 0]

        k1 = k
        k2 = k

        eta_k = np.array([[k1], [k2]])

        psi = np.array([u1, u2, v1, v2, v3, v4, v5, v6, k1, k2])

        y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + e
        rows = np.r_[psi, y]
        print("u : ", u)
        print("v : ", v)
        print("eta_k : ", eta_k)
        print("y : ", y)
        return rows

    def pls_update(self, V, Y):
        self.pls.fit(V, Y)
        return self.pls

    def plt_show1(self, n, y_act):
        plt.plot(np.arange(1, n + 1), y_act, 'bx--', lw=2, ms=10, mew=2)
        plt.xticks(np.arange(0, n, 10))
        plt.xlabel('Run No.')
        plt.ylabel('y_value')

    def plt_show2(self, n, y_act):
        plt.figure()
        plt.plot(np.arange(1, n + 1), y_act, 'ro-', linewidth=1)
        plt.xticks(np.arange(0, n, 10))
        plt.xlabel('Run No.')
        plt.ylabel('y_value')

    def plt_show3(self, n, y_act, y_pred):
        plt.plot(np.arange(1, n + 1), y_act, 'rx--', y_pred, 'bx--', linewidth=2)
        plt.xticks(np.arange(0, n, 10))
        plt.xlabel('Run No.')
        plt.ylabel('y_value')

    def DoE_Run(self, lamda_PLS, dEWMA_Wgt1, dEWMA_Wgt2, Z, M):
        N = Z * M
        I = np.identity(2)
        dEWMA_Wgt1 = dEWMA_Wgt1 * I
        dEWMA_Wgt2 = dEWMA_Wgt2 * I
        DoE_Queue = []
        uk_next = np.array([0, 0])
        Dk_prev = np.array([0, 0])
        Kd_prev = np.array([0, 0])

        vp_next = self.sampling_vp()
        ep_next = self.sampling_ep()

        for k in range(1, N + 1):  # range(101) = [0, 1, 2, ..., 100])
            result = self.sampling(k, uk_next, vp_next, ep_next, True)
            npResult = np.array(result)

            # ================================== initVM-R2R Control =====================================
            uk = npResult[0:2]
            yk = npResult[10:12]

            Dk = (yk - uk.dot(self.A)).dot(dEWMA_Wgt1) + Dk_prev.dot(I - dEWMA_Wgt1)
            Kd = (yk - uk.dot(self.A) - Dk_prev).dot(dEWMA_Wgt2) + Kd_prev.dot(I - dEWMA_Wgt2)

            Kd_prev = Kd
            Dk_prev = Dk

            uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
            print("uk_next : ", uk_next)
            vp_next = self.sampling_vp()
            ep_next = self.sampling_ep()

            DoE_Queue.append(result)

        initplsWindow = DoE_Queue.copy()
        npPlsWindow = np.array(initplsWindow)

        plsWindow = []

        for z in np.arange(0, Z):
            npPlsWindow[z * M:(z + 1) * M - 1, 0:10] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:10]
            npPlsWindow[z * M:(z + 1) * M - 1, 10:12] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, 10:12])

        for i in range(len(npPlsWindow)):
            plsWindow.append(npPlsWindow[i])

        npDoE_Queue = np.array(plsWindow)
        DoE_Mean = np.mean(npDoE_Queue, axis=0)

        plsModelData = npDoE_Queue - DoE_Mean
        V0 = plsModelData[:, 0:10]
        Y0 = plsModelData[:, 10:12]

        pls = self.pls_update(V0, Y0)

        print('Init R2R-VM Coefficients: \n', pls.coef_)

        y_pred = pls.predict(V0) + DoE_Mean[10:12]
        y_act = npDoE_Queue[:, 10:12]

        print("Init R2R-VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
        print("Init R2R-VM r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

        self.plt_show2(N, y_act[:, 0:1])

