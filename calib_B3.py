'''
calib file using for shortkeck analysis. changed from matlab version calib.m
05/08/2018
'''
import numpy as np

def calib_b3():

        calib = {}
        calib["R_SH"] = 0.0040    #Ohms (at 4K, different chip)

        calib["M_FB"] = np.array([-24.99]*16)    #Mux07-15, Mux09-8, Mux11-25
        calib["M_FB"][2] = -calib["M_FB"][2]

        calib["R_WIRE"] = np.array([151.00]*16)    #ohms include in R_FB and R_BIAS

        calib["R_BIAS"] = np.array([ 475.00, 475.00, 475.00, 475.00,
                                                475.00, 475.00, 475.00, 475.00,
                                                475.00, 475.00, 475.00, 475.00,
                                                475.00, 475.00, 475.00, 475.00])

        calib["V_B_MAX"] = np.array([5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000])

        calib["R_FB"] = np.array([2114.00]*16)

        calib["V_FB_MAX"] =  np.array([0.9720, 0.9720, 0.9720, 0.9720,
                                                        0.9720, 0.9720, 0.9720, 0.9720,
                                                        0.9720, 0.9720, 0.9720, 0.9720,
                                                        0.9720, 0.9720, 0.9720, 0.9720])

        calib["BITS_BIAS"] = 15   # number of bits in TES bias DAC
        calib["BITS_FB"] = 14   # number of bits in FB DAC

        # Bias ADC bins => bias current (Amps)
        calib["BIAS_CAL"] = (calib["V_B_MAX"]/(2**calib["BITS_BIAS"]))/(calib["R_BIAS"] + calib["R_WIRE"])
        # SQ1 FB DAC bins => TES current (Amps)
        calib["FB_CAL"] = (calib["V_FB_MAX"]/(2**calib["BITS_FB"]))/(calib["R_FB"]+calib["R_WIRE"])/calib["M_FB"]
        return calib

