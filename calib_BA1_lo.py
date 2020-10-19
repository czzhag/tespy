'''
calib file using for shortkeck analysis. changed from matlab version calib.m
05/08/2018
'''
import numpy as np

def calib_ba1():

        calib = {}
        calib["R_SH"] = 0.0030    #Ohms (at 4K, different chip)

        calib["M_FB"] = np.array([-24.99]*16)    #Mux07-15, Mux09-8, Mux11-25
        calib["M_FB"][2] = -calib["M_FB"][2]

        calib["R_WIRE"] = np.array([84.50]*16)    #ohms include in R_FB and R_BIAS

        calib["R_BIAS"] = np.array([ 400.00, 400.00, 400.00, 400.00,
                                                400.00, 400.00, 400.00, 400.00,
                                                400.00, 400.00, 400.00, 400.00,
                                                400.00, 400.00, 400.00, 400.00])

        calib["V_B_MAX"] = np.array([5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000,
                                                5.000, 5.000, 5.000, 5.000])

        calib["R_FB"] = np.array([2000.00]*16)

        calib["V_FB_MAX"] =  2*np.array([0.500, 0.500, 0.500, 0.500,
                                                        0.500, 0.500, 0.500, 0.500,
                                                        0.500, 0.500, 0.500, 0.500,
                                                        0.500, 0.500, 0.500, 0.500])

        calib["BITS_BIAS"] = 15   # number of bits in TES bias DAC
        calib["BITS_FB"] = 14   # number of bits in FB DAC

        # Bias ADC bins => bias current (Amps)
        calib["BIAS_CAL"] = (calib["V_B_MAX"]/(2**calib["BITS_BIAS"]))/(calib["R_BIAS"] + calib["R_WIRE"])
        # SQ1 FB DAC bins => TES current (Amps)
        calib["FB_CAL"] = (calib["V_FB_MAX"]/(2**calib["BITS_FB"]))/(calib["R_FB"]+calib["R_WIRE"])/calib["M_FB"]
        return calib

