import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics

class FwcSimulator:

    def __init__(self, Tgt, A, d, C, F):
        self.pls = PLSRegression(n_components=6, scale=False, max_iter=50000, copy=True)
        np.random.seed(1000000)
        self.Tgt = Tgt
        self.A = A
        self.d = d
        self.C = C
        self.F = F

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

    # def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0, 0]), ep=np.array([0, 0]), isInit=True):
    #     u1 = uk[0]
    #     u2 = uk[1]
    #     u = uk
    #
    #     v1 = vp[0]
    #     v2 = vp[1]
    #     v3 = vp[2]
    #     v4 = vp[3]
    #     v5 = vp[4]
    #     v6 = vp[5]
    #
    #     v = vp
    #     e = ep
    #
    #     k1 = k
    #     k2 = k
    #     eta_k = np.array([[k1], [k2]])
    #
    #     psi = np.array([u1, u2, v1, v2, v3, v4, v5, v6, k1, k2])
    #
    #
    #     rows = np.r_[psi, y]
    #     # print("u : ", u)
    #     # print("v : ", v)
    #     # print("eta_k : ", eta_k)
    #     # print("y : ", y)
    #     return rows

    def sampling(self, k, uk=np.array([0, 0]), vp=np.array([0, 0, 0, 0, 0, 0]), ep=np.array([0, 0]), fp=np.array([0, 0]), isInit=True):
        psi = []
        u = uk
        psi = np.r_[psi, u]
        v = vp
        psi = np.r_[psi, v]


        if fp is not None:
            f = fp

        k1 = k
        k2 = k

        eta_k = np.array(k1, k2)
        psi = np.r_[psi, eta_k]

        e = ep

        # if len(vp) > 5:
        #     psi = np.array([u1, u2, v1, v2, v3, v4, v5, v6, k1, k2])
        # else:
        #     psi = np.array([u1, u2, v1, v2, v3, v4, v5, k1, k2])

        if fp is not None:
            y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + f.dot(self.F) + e
        else:
            y = u.dot(self.A) + v.dot(self.C) + np.sum(eta_k * self.d, axis=0) + e

        rows = np.r_[psi, y]
        # print("u : ", u)
        # print("v : ", v)
        # print("eta_k : ", eta_k)
        # print("y : ", y)
        idx_end = len(rows)
        idx_start = idx_end - 2

        # print("rows : ", len(rows))
        # print("idx_end : ", idx_end)
        # print("idx_start : ", idx_start)
        return idx_start, idx_end, rows

    def pls_update(self, V, Y):
        self.pls.fit(V, Y)
        return self.pls

    def setDoE_Mean(self, DoE_Mean):
        self.DoE_Mean = DoE_Mean

    def getDoE_Mean(self):
        return self.DoE_Mean

    def setPlsWindow(self, PlsWindow):
        self.PlsWindow = PlsWindow

    def getPlsWindow(self):
        return self.PlsWindow

    def plt_show1(self, n, y_act):
#        plt.figure()
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

    def DoE_Run(self, lamda_PLS, dEWMA_Wgt1, dEWMA_Wgt2, Z, M, f, isR2R):
        N = Z * M
        I = np.identity(2)
        dEWMA_Wgt1 = dEWMA_Wgt1 * I
        dEWMA_Wgt2 = dEWMA_Wgt2 * I
        DoE_Queue = []
        uk_next = np.array([-100, 195])
        #uk_next = np.array([0, 0])
        Dk_prev = np.array([0, 0])
        Kd_prev = np.array([0, 0])

        vp_next = self.sampling_vp()
        ep_next = self.sampling_ep()

        for k in range(1, N + 1):  # range(101) = [0, 1, 2, ..., 100])
            idx_start, idx_end, result = self.sampling(k, uk_next, vp_next, ep_next, f, True)
            npResult = np.array(result)

            # ================================== initVM-R2R Control =====================================
            uk = npResult[0:2]
            yk = npResult[idx_start:idx_end]

            Dk = (yk - uk.dot(self.A)).dot(dEWMA_Wgt1) + Dk_prev.dot(I - dEWMA_Wgt1)
            Kd = (yk - uk.dot(self.A) - Dk_prev).dot(dEWMA_Wgt2) + Kd_prev.dot(I - dEWMA_Wgt2)

            Kd_prev = Kd
            Dk_prev = Dk

            if isR2R == True:
                uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                vp_next = self.sampling_vp()

            else:
                if k % M == 0:
                    uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                    vp_next = self.sampling_vp()
            ep_next = self.sampling_ep()
            DoE_Queue.append(result)

        initplsWindow = DoE_Queue.copy()
        npPlsWindow = np.array(initplsWindow)

        plsWindow = []

        for z in np.arange(0, Z):
            npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start] = lamda_PLS * npPlsWindow[z * M:(z + 1) * M - 1, 0:idx_start]
            npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end] = lamda_PLS * (npPlsWindow[z * M:(z + 1) * M - 1, idx_start:idx_end])

        for i in range(len(npPlsWindow)):
            plsWindow.append(npPlsWindow[i])

        npDoE_Queue = np.array(plsWindow)
        DoE_Mean = np.mean(npDoE_Queue, axis=0)

        plsModelData = npDoE_Queue - DoE_Mean
        V0 = plsModelData[:, 0:idx_start]
        Y0 = plsModelData[:, idx_start:idx_end]

        pls = self.pls_update(V0, Y0)

        print('Init VM Coefficients: \n', pls.coef_)

        y_pred = pls.predict(V0) + DoE_Mean[idx_start:idx_end]
        y_act = npDoE_Queue[:, idx_start:idx_end]

        # print("Init VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
        # print("Init VM r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

        self.setDoE_Mean(DoE_Mean)
        self.setPlsWindow(plsWindow)

        # if isR2R == True:
        #     self.plt_show2(N, y_act[:, 0:1])
        # else:
        #     self.plt_show1(N, y_act[:, 0:1])

    def VM_Run(self, lamda_PLS, dEWMA_Wgt1, dEWMA_Wgt2, Z, M, f, isR2R):
        N = Z * M
        I = np.identity(2)
        dEWMA_Wgt1 = dEWMA_Wgt1 * I
        dEWMA_Wgt2 = dEWMA_Wgt2 * I

        ## V0, Y0 Mean Center
        DoE_Mean = self.getDoE_Mean()
        idx_end = len(DoE_Mean)
        idx_start = idx_end - 2
        # print (len(DoE_Mean))

        meanVz = DoE_Mean[0:idx_start]
        meanYz = DoE_Mean[idx_start:idx_end]
        yk = np.array([0, 0])

        Dk_prev = np.array([0, 0])
        Kd_prev = np.array([0, 0])

        Dk = np.array([0, 0])
        Kd = np.array([0, 0])

        uk_next = np.array([-100, 195])
        # uk_next = np.array([0, 0])
        vp_next = self.sampling_vp()
        ep_next = self.sampling_ep()

        M_Queue = []
        ez_Queue = []
        ez_Queue.append([0, 0])
        y_act = []
        y_pred = []
        VM_Output = []

        plsWindow = self.getPlsWindow()

        for z in np.arange(0, Z):
            for k in np.arange(z * M + 1, ((z + 1) * M) + 1):
                idx_start, idx_end, result = self.sampling(k, uk_next, vp_next, ep_next, f, False)
                psiK = result[0:idx_start]
                psiKStar = psiK - meanVz
                y_predK = self.pls.predict(psiKStar.reshape(1, idx_start)) + meanYz
                rows = np.r_[result, y_predK.reshape(2, )]

                y_pred.append(rows[idx_end:idx_end + 2])
                y_act.append(rows[idx_start:idx_end])

                # ================================== VM + R2R Control =====================================

                if k % M != 0:
                    yk = rows[idx_end:idx_end + 2]
                else:
                    yk = rows[idx_start:idx_end]
                uk = psiK[0:2]

                Dk = (yk - uk.dot(self.A)).dot(dEWMA_Wgt1) + Dk_prev.dot(I - dEWMA_Wgt1)
                Kd = (yk - uk.dot(self.A) - Dk_prev).dot(dEWMA_Wgt2) + Kd_prev.dot(I - dEWMA_Wgt2)

                Kd_prev = Kd
                Dk_prev = Dk

                if isR2R == True:
                    uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                    vp_next = self.sampling_vp()

                # else:
                #     if k % M == 0:
                #         uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                #         vp_next = self.sampling_vp()

                uk_next = uk_next.reshape(2, )
                ep_next = self.sampling_ep()

                M_Queue.append(rows)

            del plsWindow[0:M]

            if isR2R == False:
                uk_next = (self.Tgt - Dk - Kd).dot(np.linalg.inv(self.A))
                vp_next = self.sampling_vp()

            if z == 0:
                ez = 0

            npM_Queue = np.array(M_Queue)
            npM_Queue[0:M - 1, 0:idx_start] = lamda_PLS * npM_Queue[0:M - 1, 0:idx_start]
            npM_Queue[0:M - 1, idx_start:idx_end] = lamda_PLS * (npM_Queue[0:M - 1, idx_end:idx_end + 2] + 0.5 * ez)
            npM_Queue = npM_Queue[:, 0:idx_end]

            for i in range(M):
                plsWindow.append(npM_Queue[i])

            M_Mean = np.mean(plsWindow, axis=0)
            meanVz = M_Mean[0:idx_start]
            meanYz = M_Mean[idx_start:idx_end]

            plsModelData = plsWindow - M_Mean
            V = plsModelData[:, 0:idx_start]
            Y = plsModelData[:, idx_start:idx_end]

            T = len(plsModelData)
            for i in range(T - M, T):
                VM_Output.append(Y[i])

            self.pls_update(V, Y)
            ez = M_Queue[M - 1][idx_start:idx_end] - M_Queue[M - 1][idx_end:idx_end + 2]
            ez_Queue.append(ez)

            del M_Queue[0:M]

        y_act = np.array(y_act)
        y_pred = np.array(y_pred)

        # print("VM Mean squared error: %.3f" % metrics.mean_squared_error(y_act, y_pred))
        # print("VM r2 score: %.3f" % metrics.r2_score(y_act, y_pred))

  #       if isR2R == True:
  # #          np.savetxt("process1-metrology.csv", VM_Output, delimiter=",", fmt="%s")
  #           self.plt_show2(N, y_act[:, 0:1])
  #       else:
  #           self.plt_show1(N, y_act[:, 0:1])
        VM_Output = np.array(VM_Output)
        return y_act, VM_Output

