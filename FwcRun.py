from simulator.FwcSimulator import FwcSimulator
import os
import numpy as np

Tgt = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

def main():
    os.chdir("D:/01. CLASS/Machine Learning/")
    fwc_r2r = FwcSimulator(Tgt, A_p1, d_p1, C_p1)
    fwc_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, isR2R=True)
    fwc_r2r.VM2R2R_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, isR2R=True)

    fwc_l2l = FwcSimulator(Tgt, A_p1, d_p1, C_p1)
    fwc_l2l.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, isR2R=False)
    fwc_l2l.VM2R2R_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, isR2R=False)



if __name__ == "__main__":
    main()
