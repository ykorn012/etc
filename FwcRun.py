from simulator.FwcSimulator import FwcSimulator
import os
import numpy as np

Tgt_p1 = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

Tgt_p2 = np.array([0, 0])
A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[0.05, 0], [0, 0.05]])

def main():
    os.chdir("D:/01. CLASS/Machine Learning/")
    fwc_p1_r2r = FwcSimulator(Tgt_p1, A_p1, d_p1, C_p1, None)
    fwc_p1_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=None, isR2R=True)
    # p1_r2r_y_act, p1_r2r_VMOutput = fwc_p1_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=None, isR2R=True)
    #
    # fwc_p1_l2l = FwcSimulator(Tgt_p1, A_p1, d_p1, C_p1, None)
    # fwc_p1_l2l.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=None, isR2R=False)
    # p1_l2l_y_act, p1_l2l_VMOutput = fwc_p1_l2l.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=None, isR2R=False)
    #
    # fwc_p1_r2r.plt_show2(200, p1_r2r_y_act[:, 0:1])
    # fwc_p1_l2l.plt_show1(200, p1_l2l_y_act[:, 0:1])

    # fwc_p2_r2r = FwcSimulator(Tgt_p2, A_p2, d_p2, C_p2, None)
    # fwc_p2_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=None, isR2R=True)
    # p2_r2r_y_act, p2_r2r_VMOutput = fwc_p1_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=None, isR2R=True)
    #

if __name__ == "__main__":
    main()
