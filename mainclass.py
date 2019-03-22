import re
#from sklearn.feature_extraction import FeatureHasher
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import chi2
from Bio import SeqIO
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
 
#from sklearn.multiclass import OneVsRestClassifier
 
#from sklearn.svm import LinearSVC
 
#from sklearn.pipeline import Pipeline
 
#from sklearn.feature_extraction.text import TfidfVectorizer


#from sklearn.cross_validation import cross_val_score
from Bio.SeqUtils import IsoelectricPoint
import math,copy
#from collections import OrderedDict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


###################################################################################################

##################################################quasi###################################################

## Distance is the Schneider-Wrede physicochemical distance matrix used by Chou et. al. 
_Distance1={"GW":0.923, "GV":0.464, "GT":0.272, "GS":0.158, "GR":1.0, "GQ":0.467, "GP":0.323, "GY":0.728, "GG":0.0, "GF":0.727, "GE":0.807, "GD":0.776, "GC":0.312, "GA":0.206, "GN":0.381, "GM":0.557, "GL":0.591, "GK":0.894, "GI":0.592, "GH":0.769, "ME":0.879, "MD":0.932, "MG":0.569, "MF":0.182, "MA":0.383, "MC":0.276, "MM":0.0, "ML":0.062, "MN":0.447, "MI":0.058, "MH":0.648, "MK":0.884, "MT":0.358, "MW":0.391, "MV":0.12, "MQ":0.372, "MP":0.285, "MS":0.417, "MR":1.0, "MY":0.255, "FP":0.42, "FQ":0.459, "FR":1.0, "FS":0.548, "FT":0.499, "FV":0.252, "FW":0.207, "FY":0.179, "FA":0.508, "FC":0.405, "FD":0.977, "FE":0.918, "FF":0.0, "FG":0.69, "FH":0.663, "FI":0.128, "FK":0.903, "FL":0.131, "FM":0.169, "FN":0.541, "SY":0.615, "SS":0.0, "SR":1.0, "SQ":0.358, "SP":0.181, "SW":0.827, "SV":0.342, "ST":0.174, "SK":0.883, "SI":0.478, "SH":0.718, "SN":0.289, "SM":0.44, "SL":0.474, "SC":0.185, "SA":0.1, "SG":0.17, "SF":0.622, "SE":0.812, "SD":0.801, "YI":0.23, "YH":0.678, "YK":0.904, "YM":0.268, "YL":0.219, "YN":0.512, "YA":0.587, "YC":0.478, "YE":0.932, "YD":1.0, "YG":0.782, "YF":0.202, "YY":0.0, "YQ":0.404, "YP":0.444, "YS":0.612, "YR":0.995, "YT":0.557, "YW":0.244, "YV":0.328, "LF":0.139, "LG":0.596, "LD":0.944, "LE":0.892, "LC":0.296, "LA":0.405, "LN":0.452, "LL":0.0, "LM":0.062, "LK":0.893, "LH":0.653, "LI":0.013, "LV":0.133, "LW":0.341, "LT":0.397, "LR":1.0, "LS":0.443, "LP":0.309, "LQ":0.376, "LY":0.205, "RT":0.808, "RV":0.914, "RW":1.0, "RP":0.796, "RQ":0.668, "RR":0.0, "RS":0.86, "RY":0.859, "RD":0.305, "RE":0.225, "RF":0.977, "RG":0.928, "RA":0.919, "RC":0.905, "RL":0.92, "RM":0.908, "RN":0.69, "RH":0.498, "RI":0.929, "RK":0.141, "VH":0.649, "VI":0.135, "EM":0.83, "EL":0.854, "EN":0.599, "EI":0.86, "EH":0.406, "EK":0.143, "EE":0.0, "ED":0.133, "EG":0.779, "EF":0.932, "EA":0.79, "EC":0.788, "VM":0.12, "EY":0.837, "VN":0.38, "ET":0.682, "EW":1.0, "EV":0.824, "EQ":0.598, "EP":0.688, "ES":0.726, "ER":0.234, "VP":0.212, "VQ":0.339, "VR":1.0, "VT":0.305, "VW":0.472, "KC":0.871, "KA":0.889, "KG":0.9, "KF":0.957, "KE":0.149, "KD":0.279, "KK":0.0, "KI":0.899, "KH":0.438, "KN":0.667, "KM":0.871, "KL":0.892, "KS":0.825, "KR":0.154, "KQ":0.639, "KP":0.757, "KW":1.0, "KV":0.882, "KT":0.759, "KY":0.848, "DN":0.56, "DL":0.841, "DM":0.819, "DK":0.249, "DH":0.435, "DI":0.847, "DF":0.924, "DG":0.697, "DD":0.0, "DE":0.124, "DC":0.742, "DA":0.729, "DY":0.836, "DV":0.797, "DW":1.0, "DT":0.649, "DR":0.295, "DS":0.667, "DP":0.657, "DQ":0.584, "QQ":0.0, "QP":0.272, "QS":0.461, "QR":1.0, "QT":0.389, "QW":0.831, "QV":0.464, "QY":0.522, "QA":0.512, "QC":0.462, "QE":0.861, "QD":0.903, "QG":0.648, "QF":0.671, "QI":0.532, "QH":0.765, "QK":0.881, "QM":0.505, "QL":0.518, "QN":0.181, "WG":0.829, "WF":0.196, "WE":0.931, "WD":1.0, "WC":0.56, "WA":0.658, "WN":0.631, "WM":0.344, "WL":0.304, "WK":0.892, "WI":0.305, "WH":0.678, "WW":0.0, "WV":0.418, "WT":0.638, "WS":0.689, "WR":0.968, "WQ":0.538, "WP":0.555, "WY":0.204, "PR":1.0, "PS":0.196, "PP":0.0, "PQ":0.228, "PV":0.244, "PW":0.72, "PT":0.161, "PY":0.481, "PC":0.179, "PA":0.22, "PF":0.515, "PG":0.376, "PD":0.852, "PE":0.831, "PK":0.875, "PH":0.696, "PI":0.363, "PN":0.231, "PL":0.357, "PM":0.326, "CK":0.887, "CI":0.304, "CH":0.66, "CN":0.324, "CM":0.277, "CL":0.301, "CC":0.0, "CA":0.114, "CG":0.32, "CF":0.437, "CE":0.838, "CD":0.847, "CY":0.457, "CS":0.176, "CR":1.0, "CQ":0.341, "CP":0.157, "CW":0.639, "CV":0.167, "CT":0.233, "IY":0.213, "VA":0.275, "VC":0.165, "VD":0.9, "VE":0.867, "VF":0.269, "VG":0.471, "IQ":0.383, "IP":0.311, "IS":0.443, "IR":1.0, "VL":0.134, "IT":0.396, "IW":0.339, "IV":0.133, "II":0.0, "IH":0.652, "IK":0.892, "VS":0.322, "IM":0.057, "IL":0.013, "VV":0.0, "IN":0.457, "IA":0.403, "VY":0.31, "IC":0.296, "IE":0.891, "ID":0.942, "IG":0.592, "IF":0.134, "HY":0.821, "HR":0.697, "HS":0.865, "HP":0.777, "HQ":0.716, "HV":0.831, "HW":0.981, "HT":0.834, "HK":0.566, "HH":0.0, "HI":0.848, "HN":0.754, "HL":0.842, "HM":0.825, "HC":0.836, "HA":0.896, "HF":0.907, "HG":1.0, "HD":0.629, "HE":0.547, "NH":0.78, "NI":0.615, "NK":0.891, "NL":0.603, "NM":0.588, "NN":0.0, "NA":0.424, "NC":0.425, "ND":0.838, "NE":0.835, "NF":0.766, "NG":0.512, "NY":0.641, "NP":0.266, "NQ":0.175, "NR":1.0, "NS":0.361, "NT":0.368, "NV":0.503, "NW":0.945, "TY":0.596, "TV":0.345, "TW":0.816, "TT":0.0, "TR":1.0, "TS":0.185, "TP":0.159, "TQ":0.322, "TN":0.315, "TL":0.453, "TM":0.403, "TK":0.866, "TH":0.737, "TI":0.455, "TF":0.604, "TG":0.312, "TD":0.83, "TE":0.812, "TC":0.261, "TA":0.251, "AA":0.0, "AC":0.112, "AE":0.827, "AD":0.819, "AG":0.208, "AF":0.54, "AI":0.407, "AH":0.696, "AK":0.891, "AM":0.379, "AL":0.406, "AN":0.318, "AQ":0.372, "AP":0.191, "AS":0.094, "AR":1.0, "AT":0.22, "AW":0.739, "AV":0.273, "AY":0.552, "VK":0.889 }

## Distance is the Grantham chemical distance matrix used by Grantham et. al. 
_Distance2={"GW":0.923, "GV":0.464, "GT":0.272, "GS":0.158, "GR":1.0, "GQ":0.467, "GP":0.323, "GY":0.728, "GG":0.0, "GF":0.727, "GE":0.807, "GD":0.776, "GC":0.312, "GA":0.206, "GN":0.381, "GM":0.557, "GL":0.591, "GK":0.894, "GI":0.592, "GH":0.769, "ME":0.879, "MD":0.932, "MG":0.569, "MF":0.182, "MA":0.383, "MC":0.276, "MM":0.0, "ML":0.062, "MN":0.447, "MI":0.058, "MH":0.648, "MK":0.884, "MT":0.358, "MW":0.391, "MV":0.12, "MQ":0.372, "MP":0.285, "MS":0.417, "MR":1.0, "MY":0.255, "FP":0.42, "FQ":0.459, "FR":1.0, "FS":0.548, "FT":0.499, "FV":0.252, "FW":0.207, "FY":0.179, "FA":0.508, "FC":0.405, "FD":0.977, "FE":0.918, "FF":0.0, "FG":0.69, "FH":0.663, "FI":0.128, "FK":0.903, "FL":0.131, "FM":0.169, "FN":0.541, "SY":0.615, "SS":0.0, "SR":1.0, "SQ":0.358, "SP":0.181, "SW":0.827, "SV":0.342, "ST":0.174, "SK":0.883, "SI":0.478, "SH":0.718, "SN":0.289, "SM":0.44, "SL":0.474, "SC":0.185, "SA":0.1, "SG":0.17, "SF":0.622, "SE":0.812, "SD":0.801, "YI":0.23, "YH":0.678, "YK":0.904, "YM":0.268, "YL":0.219, "YN":0.512, "YA":0.587, "YC":0.478, "YE":0.932, "YD":1.0, "YG":0.782, "YF":0.202, "YY":0.0, "YQ":0.404, "YP":0.444, "YS":0.612, "YR":0.995, "YT":0.557, "YW":0.244, "YV":0.328, "LF":0.139, "LG":0.596, "LD":0.944, "LE":0.892, "LC":0.296, "LA":0.405, "LN":0.452, "LL":0.0, "LM":0.062, "LK":0.893, "LH":0.653, "LI":0.013, "LV":0.133, "LW":0.341, "LT":0.397, "LR":1.0, "LS":0.443, "LP":0.309, "LQ":0.376, "LY":0.205, "RT":0.808, "RV":0.914, "RW":1.0, "RP":0.796, "RQ":0.668, "RR":0.0, "RS":0.86, "RY":0.859, "RD":0.305, "RE":0.225, "RF":0.977, "RG":0.928, "RA":0.919, "RC":0.905, "RL":0.92, "RM":0.908, "RN":0.69, "RH":0.498, "RI":0.929, "RK":0.141, "VH":0.649, "VI":0.135, "EM":0.83, "EL":0.854, "EN":0.599, "EI":0.86, "EH":0.406, "EK":0.143, "EE":0.0, "ED":0.133, "EG":0.779, "EF":0.932, "EA":0.79, "EC":0.788, "VM":0.12, "EY":0.837, "VN":0.38, "ET":0.682, "EW":1.0, "EV":0.824, "EQ":0.598, "EP":0.688, "ES":0.726, "ER":0.234, "VP":0.212, "VQ":0.339, "VR":1.0, "VT":0.305, "VW":0.472, "KC":0.871, "KA":0.889, "KG":0.9, "KF":0.957, "KE":0.149, "KD":0.279, "KK":0.0, "KI":0.899, "KH":0.438, "KN":0.667, "KM":0.871, "KL":0.892, "KS":0.825, "KR":0.154, "KQ":0.639, "KP":0.757, "KW":1.0, "KV":0.882, "KT":0.759, "KY":0.848, "DN":0.56, "DL":0.841, "DM":0.819, "DK":0.249, "DH":0.435, "DI":0.847, "DF":0.924, "DG":0.697, "DD":0.0, "DE":0.124, "DC":0.742, "DA":0.729, "DY":0.836, "DV":0.797, "DW":1.0, "DT":0.649, "DR":0.295, "DS":0.667, "DP":0.657, "DQ":0.584, "QQ":0.0, "QP":0.272, "QS":0.461, "QR":1.0, "QT":0.389, "QW":0.831, "QV":0.464, "QY":0.522, "QA":0.512, "QC":0.462, "QE":0.861, "QD":0.903, "QG":0.648, "QF":0.671, "QI":0.532, "QH":0.765, "QK":0.881, "QM":0.505, "QL":0.518, "QN":0.181, "WG":0.829, "WF":0.196, "WE":0.931, "WD":1.0, "WC":0.56, "WA":0.658, "WN":0.631, "WM":0.344, "WL":0.304, "WK":0.892, "WI":0.305, "WH":0.678, "WW":0.0, "WV":0.418, "WT":0.638, "WS":0.689, "WR":0.968, "WQ":0.538, "WP":0.555, "WY":0.204, "PR":1.0, "PS":0.196, "PP":0.0, "PQ":0.228, "PV":0.244, "PW":0.72, "PT":0.161, "PY":0.481, "PC":0.179, "PA":0.22, "PF":0.515, "PG":0.376, "PD":0.852, "PE":0.831, "PK":0.875, "PH":0.696, "PI":0.363, "PN":0.231, "PL":0.357, "PM":0.326, "CK":0.887, "CI":0.304, "CH":0.66, "CN":0.324, "CM":0.277, "CL":0.301, "CC":0.0, "CA":0.114, "CG":0.32, "CF":0.437, "CE":0.838, "CD":0.847, "CY":0.457, "CS":0.176, "CR":1.0, "CQ":0.341, "CP":0.157, "CW":0.639, "CV":0.167, "CT":0.233, "IY":0.213, "VA":0.275, "VC":0.165, "VD":0.9, "VE":0.867, "VF":0.269, "VG":0.471, "IQ":0.383, "IP":0.311, "IS":0.443, "IR":1.0, "VL":0.134, "IT":0.396, "IW":0.339, "IV":0.133, "II":0.0, "IH":0.652, "IK":0.892, "VS":0.322, "IM":0.057, "IL":0.013, "VV":0.0, "IN":0.457, "IA":0.403, "VY":0.31, "IC":0.296, "IE":0.891, "ID":0.942, "IG":0.592, "IF":0.134, "HY":0.821, "HR":0.697, "HS":0.865, "HP":0.777, "HQ":0.716, "HV":0.831, "HW":0.981, "HT":0.834, "HK":0.566, "HH":0.0, "HI":0.848, "HN":0.754, "HL":0.842, "HM":0.825, "HC":0.836, "HA":0.896, "HF":0.907, "HG":1.0, "HD":0.629, "HE":0.547, "NH":0.78, "NI":0.615, "NK":0.891, "NL":0.603, "NM":0.588, "NN":0.0, "NA":0.424, "NC":0.425, "ND":0.838, "NE":0.835, "NF":0.766, "NG":0.512, "NY":0.641, "NP":0.266, "NQ":0.175, "NR":1.0, "NS":0.361, "NT":0.368, "NV":0.503, "NW":0.945, "TY":0.596, "TV":0.345, "TW":0.816, "TT":0.0, "TR":1.0, "TS":0.185, "TP":0.159, "TQ":0.322, "TN":0.315, "TL":0.453, "TM":0.403, "TK":0.866, "TH":0.737, "TI":0.455, "TF":0.604, "TG":0.312, "TD":0.83, "TE":0.812, "TC":0.261, "TA":0.251, "AA":0.0, "AC":0.112, "AE":0.827, "AD":0.819, "AG":0.208, "AF":0.54, "AI":0.407, "AH":0.696, "AK":0.891, "AM":0.379, "AL":0.406, "AN":0.318, "AQ":0.372, "AP":0.191, "AS":0.094, "AR":1.0, "AT":0.22, "AW":0.739, "AV":0.273, "AY":0.552, "VK":0.889 }

def GetSequenceOrderCouplingNumber(ProteinSequence,d=1,distancematrix=_Distance1):
	"""
	Computing the dth-rank sequence order coupling number for a protein.
	d is the gap between two amino acids.
	Output: result is numeric value.
	"""
    
	NumProtein=len(ProteinSequence)
	tau=0.0
	for i in range(NumProtein-d):
		temp1=ProteinSequence[i]
		temp2=ProteinSequence[i+d]
		tau=tau+math.pow(distancematrix[temp1+temp2],2)
	return round(tau,3)
#############################################################################################
def GetSequenceOrderCouplingNumberp(ProteinSequence,maxlag=30,distancematrix={}):
	"""
	Computing the sequence order coupling numbers from 1 to maxlag for a given protein sequence based on the user-defined property.

	maxlag is the maximum lag and the length of the protein should be larger than maxlag. default is 30.
	
	distancematrix is the a dict form containing 400 distance values
    result:all sequence order coupling numbers based on the given property
	"""
	NumProtein=len(ProteinSequence)
	Tau={}
	for i in range(maxlag):
		Tau["tau"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau

#############################################################################################
    

def GetSequenceOrderCouplingNumberSW(ProteinSequence,maxlag=30,distancematrix=_Distance1):
	"""

	Computing the sequence order coupling numbers from 1 to maxlag for a given protein sequence based on the Schneider-Wrede physicochemical distance matrix
    
	distancematrix is a dict form containing Schneider-Wrede physicochemical 
	
	Output: result is a dict form containing all sequence order coupling numbers based
	
	on the Schneider-Wrede physicochemical distance matrix
	"""
    
	NumProtein=len(ProteinSequence)
	Tau={}
	for i in range(maxlag):
		Tau["tausw"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau

#############################################################################################
def GetSequenceOrderCouplingNumberGrant(ProteinSequence,maxlag=30,distancematrix=_Distance2):
	"""
	Computing the sequence order coupling numbers from 1 to maxlag for a given protein sequence based on the Grantham chemical distance matrix. 
    
	distancematrix is a dict form containing Grantham chemical distance matrix. omitted!
	
	Output: result is a dict form containing all sequence order coupling numbers based on the Grantham chemical distance matrix
	"""
	NumProtein=len(ProteinSequence)
	Tau={}
	for i in range(maxlag):
		Tau["taugrant"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau

#########################################################################################################
    
def GetSequenceOrderCouplingNumberTotal(ProteinSequence,maxlag=30):
	"""
	Computing the sequence order coupling numbers from 1 to maxlag for a given protein sequence.
    result is a dict form containing all sequence order coupling numbers
	"""
    
	Tau={}
	Tau.update(GetSequenceOrderCouplingNumberSW(ProteinSequence,maxlag=maxlag))
	Tau.update(GetSequenceOrderCouplingNumberGrant(ProteinSequence,maxlag=maxlag))
	return Tau



##########################################################################################################
    
def GetAAComposition(ProteinSequence):

	"""
	Calculate the composition of Amino acids for a given protein sequence.
	Output: result is a dict form containing the composition of 20 amino acids.
	"""
    
    
	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		Result[i]=round(float(ProteinSequence.count(i))/LengthSequence,3)
	return Result

########################################################################################################
    

def GetQuasiSequenceOrder1(ProteinSequence,maxlag=30,weight=0.1,distancematrix={}):
	"""
	Computing the first 20 quasi-sequence-order descriptors for a given protein sequence.

	see method GetQuasiSequenceOrder for the choice of parameters.
	"""
    
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSO'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
    


def GetQuasiSequenceOrder2(ProteinSequence,maxlag=30,weight=0.1,distancematrix={}):
	"""
	Computing the last maxlag quasi-sequence-order descriptors for a given protein sequence.
	
	method GetQuasiSequenceOrder for the choice of parameters.
	"""
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSO'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result


#############################################################################################
    

def GetQuasiSequenceOrder1SW(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance1):
	"""
	Computing the first 20 quasi-sequence-order descriptors for a given protein sequence.

	method GetQuasiSequenceOrder for the choice of parameters.
	"""
    
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSOSW'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
    

def GetQuasiSequenceOrder2SW(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance1):
	"""
	Computing the last maxlag quasi-sequence-order descriptors for a given protein sequence.
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	"""
    
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSOSW'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result

###############################################################################################
    

def GetQuasiSequenceOrder1Grant(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance2):
	"""
	Computing the first 20 quasi-sequence-order descriptors for a given protein sequence.
   method GetQuasiSequenceOrder for the choice of parameters.
	"""
    
    
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSOgrant'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
    

def GetQuasiSequenceOrder2Grant(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance2):
	"""
	Computing the last maxlag quasi-sequence-order descriptors for a given protein sequence.
	
	method GetQuasiSequenceOrder for the choice of parameters.
	"""
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSOgrant'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result


#############################################################################################
    
    
def GetQuasiSequenceOrder(ProteinSequence,maxlag=30,weight=0.1):

	result=dict()
	result.update(GetQuasiSequenceOrder1SW(ProteinSequence,maxlag,weight,_Distance1))
	result.update(GetQuasiSequenceOrder2SW(ProteinSequence,maxlag,weight,_Distance1))
	result.update(GetQuasiSequenceOrder1Grant(ProteinSequence,maxlag,weight,_Distance2))
	result.update(GetQuasiSequenceOrder2Grant(ProteinSequence,maxlag,weight,_Distance2))
	return result


#############################################################################################
    

def GetQuasiSequenceOrderp(ProteinSequence,maxlag=30,weight=0.1,distancematrix={}):
	"""
	weight is a weight factor.  please see reference 1 for its choice. default is 0.1.
    distancematrix is a dict form containing 400 distance values
	Output: result is a dict form containing all quasi-sequence-order descriptors
	"""
	result=dict()
	result.update(GetQuasiSequenceOrder1(ProteinSequence,maxlag,weight,distancematrix))
	result.update(GetQuasiSequenceOrder2(ProteinSequence,maxlag,weight,distancematrix))
	return result



##################################################CTD#######################################################################
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

_Hydrophobicity={'1':'RKEDQN','2':'GASTPHY','3':'CLVIMFW'}  #'1'stand for Polar; '2'stand for Neutral, '3' stand for Hydrophobicity

_NormalizedVDWV={'1':'GASTPD','2':'NVEQIL','3':'MHKFRYW'} #'1'stand for (0-2.78); '2'stand for (2.95-4.0), '3' stand for (4.03-8.08)

_Polarity={'1':'LIFWCMVY','2':'CPNVEQIL','3':'KMHFRYW'} #'1'stand for (4.9-6.2); '2'stand for (8.0-9.2), '3' stand for (10.4-13.0)

_Charge={'1':'KR','2':'ANCQGHILMFPSTWYV','3':'DE'} #'1'stand for Positive; '2'stand for Neutral, '3' stand for Negative

_SecondaryStr={'1':'EALMQKRH','2':'VIYCWFT','3':'GNPSD'} #'1'stand for Helix; '2'stand for Strand, '3' stand for coil

_SolventAccessibility={'1':'ALFCGIVW','2':'RKQEND','3':'MPSTHY'} #'1'stand for Buried; '2'stand for Exposed, '3' stand for Intermediate

_Polarizability={'1':'GASDT','2':'CPNVEQIL','3':'KMHFRYW'} #'1'stand for (0-0.108); '2'stand for (0.128-0.186), '3' stand for (0.219-0.409)


##You can continuely add other properties of AADs to compute descriptors of protein sequence.

_AATProperty=(_Hydrophobicity,_NormalizedVDWV,_Polarity,_Charge,_SecondaryStr,_SolventAccessibility,_Polarizability)

_AATPropertyName=('_Hydrophobicity','_NormalizedVDWV','_Polarity','_Charge','_SecondaryStr','_SolventAccessibility','_Polarizability')


##################################################################################################

def StringtoNum(ProteinSequence,AAProperty): 
	"""
	Tranform the protein sequence into the string form such as 32123223132121123.
	AAProperty is a dict form containing classifciation of amino acids such as _Polarizability. result is a string such as 123321222132111123222
	"""
	
	hardProteinSequence=copy.deepcopy(ProteinSequence)
	for k,m in AAProperty.items():
		for index in str(m):
			hardProteinSequence=str.replace(hardProteinSequence,index,k)
	TProteinSequence=hardProteinSequence

	return TProteinSequence


def CalculateComposition(ProteinSequence,AAProperty,AAPName):
	"""
	A method used for computing composition descriptors.
	AAProperty is a dict form containing classifciation of amino acids such as _Polarizability.	
	AAPName is a string used for indicating a AAP name.	
    result is a dict form containing composition descriptors based on the given property.
	"""
	TProteinSequence=StringtoNum(ProteinSequence,AAProperty)
	Result={}
	Num=len(TProteinSequence)
	Result[AAPName+'C'+'1']=round(float(TProteinSequence.count('1'))/Num,3)
	Result[AAPName+'C'+'2']=round(float(TProteinSequence.count('2'))/Num,3)
	Result[AAPName+'C'+'3']=round(float(TProteinSequence.count('3'))/Num,3)
	return Result

def CalculateTransition(ProteinSequence,AAProperty,AAPName):  
	"""
	A method used for computing transition descriptors	
	AAProperty is a dict form containing classifciation of amino acids such as _Polarizability.
    
	AAPName is a string used for indicating a AAP name.
    
	Output:result is a dict form containing transition descriptors based on the given property.
	"""
	
	TProteinSequence=StringtoNum(ProteinSequence,AAProperty)
	Result={}
	Num=len(TProteinSequence)
	CTD=TProteinSequence
	Result[AAPName+'T'+'12']=round(float(CTD.count('12')+CTD.count('21'))/(Num-1),3)
	Result[AAPName+'T'+'13']=round(float(CTD.count('13')+CTD.count('31'))/(Num-1),3)
	Result[AAPName+'T'+'23']=round(float(CTD.count('23')+CTD.count('32'))/(Num-1),3)
	return Result



def CalculateDistribution(ProteinSequence,AAProperty,AAPName):  
	
	"""
	A method used for computing distribution descriptors.
	"""
	TProteinSequence=StringtoNum(ProteinSequence,AAProperty)
	Result={}
	Num=len(TProteinSequence)
	temp=('1','2','3')
	for i in temp:
		num=TProteinSequence.count(i)
		ink=1
		indexk=0
		cds=[]
		while ink<=num:
			indexk=str.find(TProteinSequence,i,indexk)+1
			cds.append(indexk)
			ink=ink+1
				
		if cds==[]:
			Result[AAPName+'D'+i+'001']=0
			Result[AAPName+'D'+i+'025']=0
			Result[AAPName+'D'+i+'050']=0
			Result[AAPName+'D'+i+'075']=0
			Result[AAPName+'D'+i+'100']=0
		else:
				
			Result[AAPName+'D'+i+'001']=round(float(cds[0])/Num*100,3)
			Result[AAPName+'D'+i+'025']=round(float(cds[int(math.floor(num*0.25))-1])/Num*100,3)
			Result[AAPName+'D'+i+'050']=round(float(cds[int(math.floor(num*0.5))-1])/Num*100,3)
			Result[AAPName+'D'+i+'075']=round(float(cds[int(math.floor(num*0.75))-1])/Num*100,3)
			Result[AAPName+'D'+i+'100']=round(float(cds[-1])/Num*100,3)

	return Result

##################################################################################################
def CalculateCompositionHydrophobicity(ProteinSequence):
	
	"""
	A method used for calculating composition descriptors based on Hydrophobicity of AADs.
	"""
	
	result=CalculateComposition(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	
def CalculateCompositionNormalizedVDWV(ProteinSequence):
	"""
	A method used for calculating composition descriptors based on NormalizedVDWV of AADs.
	"""
	result=CalculateComposition(ProteinSequence,_NormalizedVDWV,'_NormalizedVDWV')
	return result
	
def CalculateCompositionPolarity(ProteinSequence):
	"""
	A method used for calculating composition descriptors based on Polarity of AADs.	
	"""
	
	result=CalculateComposition(ProteinSequence,_Polarity,'_Polarity')
	return result
	
def CalculateCompositionCharge(ProteinSequence):
	"""
	A method used for calculating composition descriptors based on Charge of AADs.

	"""
	
	result=CalculateComposition(ProteinSequence,_Charge,'_Charge')
	return result
	
def CalculateCompositionSecondaryStr(ProteinSequence):
	"""
	###############################################################################################
	A method used for calculating composition descriptors based on SecondaryStr of AADs.
	"""
	
	result=CalculateComposition(ProteinSequence,_SecondaryStr,'_SecondaryStr')
	return result
	
def CalculateCompositionSolventAccessibility(ProteinSequence):
	"""
	A method used for calculating composition descriptors based on SolventAccessibility of  AADs.	
	Output:result is a dict form containing Composition descriptors based on SolventAccessibility.
	"""
	
	result=CalculateComposition(ProteinSequence,_SolventAccessibility,'_SolventAccessibility')
	return result
##################################################################################################
def CalculateCompositionPolarizability(ProteinSequence):
	"""
	A method used for calculating composition descriptors based on Polarizability of AADs.
	"""
	
	result=CalculateComposition(ProteinSequence,_Polarizability,'_Polarizability')
	return result




##################################################################################################
def CalculateTransitionHydrophobicity(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on Hydrophobicity ofAADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	
def CalculateTransitionNormalizedVDWV(ProteinSequence):
	"""
	###############################################################################################
	A method used for calculating Transition descriptors based on NormalizedVDWV of AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_NormalizedVDWV,'_NormalizedVDWV')
	return result
	
def CalculateTransitionPolarity(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on Polarity of AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_Polarity,'_Polarity')
	return result
	
def CalculateTransitionCharge(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on Charge of AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_Charge,'_Charge')
	return result
	
def CalculateTransitionSecondaryStr(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on SecondaryStr of AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_SecondaryStr,'_SecondaryStr')
	return result
	
def CalculateTransitionSolventAccessibility(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on SolventAccessibility of  AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_SolventAccessibility,'_SolventAccessibility')
	return result
	
def CalculateTransitionPolarizability(ProteinSequence):
	"""
	A method used for calculating Transition descriptors based on Polarizability of AADs.
	"""
	
	result=CalculateTransition(ProteinSequence,_Polarizability,'_Polarizability')
	return result

##################################################################################################
##################################################################################################
def CalculateDistributionHydrophobicity(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on Hydrophobicity of AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	
def CalculateDistributionNormalizedVDWV(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on NormalizedVDWV of AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_NormalizedVDWV,'_NormalizedVDWV')
	return result
	
def CalculateDistributionPolarity(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on Polarity of AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_Polarity,'_Polarity')
	return result
	
def CalculateDistributionCharge(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on Charge of ADDs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_Charge,'_Charge')
	return result
	
def CalculateDistributionSecondaryStr(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on SecondaryStr of AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_SecondaryStr,'_SecondaryStr')
	return result
	
def CalculateDistributionSolventAccessibility(ProteinSequence):
	
	"""
	A method used for calculating Distribution descriptors based on SolventAccessibility of  AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_SolventAccessibility,'_SolventAccessibility')
	return result
	
def CalculateDistributionPolarizability(ProteinSequence):
	"""
	A method used for calculating Distribution descriptors based on Polarizability of AADs.
	"""
	
	result=CalculateDistribution(ProteinSequence,_Polarizability,'_Polarizability')
	return result

##################################################################################################

def CalculateC(ProteinSequence):
	"""
	Calculate all composition descriptors based seven different properties of AADs.
	"""
	result={}
	result.update(CalculateCompositionPolarizability(ProteinSequence))
	result.update(CalculateCompositionSolventAccessibility(ProteinSequence))
	result.update(CalculateCompositionSecondaryStr(ProteinSequence))
	result.update(CalculateCompositionCharge(ProteinSequence))
	result.update(CalculateCompositionPolarity(ProteinSequence))
	result.update(CalculateCompositionNormalizedVDWV(ProteinSequence))
	result.update(CalculateCompositionHydrophobicity(ProteinSequence))
	return result
	
def CalculateT(ProteinSequence):
	"""
	Calculate all transition descriptors based seven different properties of AADs.
	"""
	result={}
	result.update(CalculateTransitionPolarizability(ProteinSequence))
	result.update(CalculateTransitionSolventAccessibility(ProteinSequence))
	result.update(CalculateTransitionSecondaryStr(ProteinSequence))
	result.update(CalculateTransitionCharge(ProteinSequence))
	result.update(CalculateTransitionPolarity(ProteinSequence))
	result.update(CalculateTransitionNormalizedVDWV(ProteinSequence))
	result.update(CalculateTransitionHydrophobicity(ProteinSequence))
	return result
	
def CalculateD(ProteinSequence):
	"""
	Calculate all distribution descriptors based seven different properties of AADs.
	"""
	result={}
	result.update(CalculateDistributionPolarizability(ProteinSequence))
	result.update(CalculateDistributionSolventAccessibility(ProteinSequence))
	result.update(CalculateDistributionSecondaryStr(ProteinSequence))
	result.update(CalculateDistributionCharge(ProteinSequence))
	result.update(CalculateDistributionPolarity(ProteinSequence))
	result.update(CalculateDistributionNormalizedVDWV(ProteinSequence))
	result.update(CalculateDistributionHydrophobicity(ProteinSequence))
	return result


def CalculateCTD(ProteinSequence):
	"""
	Calculate all CTD descriptors based seven different properties of AADs.
	"""
	result={}
	result.update(CalculateCompositionPolarizability(ProteinSequence))
	result.update(CalculateCompositionSolventAccessibility(ProteinSequence))
	result.update(CalculateCompositionSecondaryStr(ProteinSequence))
	result.update(CalculateCompositionCharge(ProteinSequence))
	result.update(CalculateCompositionPolarity(ProteinSequence))
	result.update(CalculateCompositionNormalizedVDWV(ProteinSequence))
	result.update(CalculateCompositionHydrophobicity(ProteinSequence))
	result.update(CalculateTransitionPolarizability(ProteinSequence))
	result.update(CalculateTransitionSolventAccessibility(ProteinSequence))
	result.update(CalculateTransitionSecondaryStr(ProteinSequence))
	result.update(CalculateTransitionCharge(ProteinSequence))
	result.update(CalculateTransitionPolarity(ProteinSequence))
	result.update(CalculateTransitionNormalizedVDWV(ProteinSequence))
	result.update(CalculateTransitionHydrophobicity(ProteinSequence))
	result.update(CalculateDistributionPolarizability(ProteinSequence))
	result.update(CalculateDistributionSolventAccessibility(ProteinSequence))
	result.update(CalculateDistributionSecondaryStr(ProteinSequence))
	result.update(CalculateDistributionCharge(ProteinSequence))
	result.update(CalculateDistributionPolarity(ProteinSequence))
	result.update(CalculateDistributionNormalizedVDWV(ProteinSequence))
	result.update(CalculateDistributionHydrophobicity(ProteinSequence))
	return result


##################################################################################################



###############################################Conjoint Tried method #############################################

#a Dipole scale (Debye): -, Dipole<1.0; +, 1.0<Dipole<2.0; ++, 2.0<Dipole<3.0; +++, Dipole>3.0; +'+'+', Dipole>3.0 with opposite orientation.
#b Volume scale (Ã…3): -, Volume<50; +, Volume> 50.
#c Cys is separated from class 3 because of its ability to form disulfide bonds.
 
_repmat={1:["A",'G','V'],2:['I','L','F','P'],3:['Y','M','T','S'],4:['H','N','Q','W'],5:['R','K'],6:['D','E'],7:['C']}
def _Str2Num(proteinsequence):
	"""
	translate the amino acid letter into the corresponding class based on the
	
	given form.
	
	"""
	repmat={}
	for i in _repmat:
		for j in _repmat[i]:
			repmat[j]=i
			
	res=proteinsequence
	for i in repmat:
		res=res.replace(i,str(repmat[i]))
	return res


###############################################################################
def CalculateConjointTriad(proteinsequence):

    res={}
    proteinnum=_Str2Num(proteinsequence)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                temp=str(i)+str(j)+str(k)
                res[temp]=proteinnum.count(temp)
    return res

###############################################################################################################
_Hydrophobicity={"A":0.02,"R":-0.42,"N":-0.77,"D":-1.04,"C":0.77,"Q":-1.10,"E":-1.14,"G":-0.80,"H":0.26,"I":1.81,"L":1.14,"K":-0.41,"M":1.00,"F":1.35,"P":-0.09,"S":-0.97,"T":-0.77,"W":1.71,"Y":1.11,"V":1.13}

_AvFlexibility={"A":0.357,"R":0.529,"N":0.463,"D":0.511,"C":0.346,"Q":0.493,"E":0.497,"G":0.544,"H":0.323,"I":0.462,"L":0.365,"K":0.466,"M":0.295,"F":0.314,"P":0.509,"S":0.507,"T":0.444,"W":0.305,"Y":0.420,"V":0.386}

_Polarizability={"A":0.046,"R":0.291,"N":0.134,"D":0.105,"C":0.128,"Q":0.180,"E":0.151,"G":0.000,"H":0.230,"I":0.186,"L":0.186,"K":0.219,"M":0.221,"F":0.290,"P":0.131,"S":0.062,"T":0.108,"W":0.409,"Y":0.298,"V":0.140}

_FreeEnergy={"A":-0.368,"R":-1.03,"N":0.0,"D":2.06,"C":4.53,"Q":0.731,"E":1.77,"G":-0.525,"H":0.0,"I":0.791,"L":1.07,"K":0.0,"M":0.656,"F":1.06,"P":-2.24,"S":-0.524,"T":0.0,"W":1.60,"Y":4.91,"V":0.401}

_ResidueASA={"A":115.0,"R":225.0,"N":160.0,"D":150.0,"C":135.0,"Q":180.0,"E":190.0,"G":75.0,"H":195.0,"I":175.0,"L":170.0,"K":200.0,"M":185.0,"F":210.0,"P":145.0,"S":115.0,"T":140.0,"W":255.0,"Y":230.0,"V":155.0}

_ResidueVol={"A":52.6,"R":109.1,"N":75.7,"D":68.4,"C":68.3,"Q":89.7,"E":84.7,"G":36.3,"H":91.9,"I":102.0,"L":102.0,"K":105.1,"M":97.7,"F":113.9,"P":73.6,"S":54.9,"T":71.2,"W":135.4,"Y":116.2,"V":85.1}

_Steric={"A":0.52,"R":0.68,"N":0.76,"D":0.76,"C":0.62,"Q":0.68,"E":0.68,"G":0.00,"H":0.70,"I":1.02,"L":0.98,"K":0.68,"M":0.78,"F":0.70,"P":0.36,"S":0.53,"T":0.50,"W":0.70,"Y":0.70,"V":0.76}

_Mutability={"A":100.0,"R":65.0,"N":134.0,"D":106.0,"C":20.0,"Q":93.0,"E":102.0,"G":49.0,"H":66.0,"I":96.0,"L":40.0,"K":-56.0,"M":94.0,"F":41.0,"P":56.0,"S":120.0,"T":97.0,"W":18.0,"Y":41.0,"V":74.0}


_AAProperty=(_Hydrophobicity,_AvFlexibility,_Polarizability,_FreeEnergy,_ResidueASA,_ResidueVol,_Steric,_Mutability)

_AAPropertyName=('_Hydrophobicity','_AvFlexibility','_Polarizability','_FreeEnergy','_ResidueASA','_ResidueVol','_Steric','_Mutability')			 

##################################################################################################
def _mean(listvalue):
	"""
	The mean value of the list data.
	"""
	return sum(listvalue)/len(listvalue)
##################################################################################################
def _std(listvalue,ddof=1): #The standard deviation of the list data.
    mean=_mean(listvalue)
    
    temp=[math.pow(i-mean,2) for i in listvalue]
    res=math.sqrt(sum(temp)/(len(listvalue)-ddof))
    return res
##################################################################################################

def NormalizeEachAAP(AAP):
	"""
	
	Input: AAP is a dict form containing the properties of 20 amino acids.
	
	Output: result is the a dict form containing the normalized properties 
	
	of 20 amino acids.
	####################################################################################
	"""
	if len(AAP.values())!=20:
		print( 'You can not input the correct number of properities of Amino acids!')
	else:
		Result={}
		for i,j in AAP.items():
			Result[i]=(j-_mean(AAP.values()))/_std(AAP.values(),ddof=0)

	return Result

def CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,AAP,AAPName):
	"""
	you can use the function to compute MoreauBrotoAuto descriptors for different properties based on AADs.
	Input: protein is a pure protein sequence.
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	Output: result is a dict form containing 30 Normalized Moreau-Broto autocorrelation descriptors based on the given property.
	"""
		
	AAPdic=NormalizeEachAAP(AAP)

	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
			temp=temp+AAPdic[ProteinSequence[j]]*AAPdic[ProteinSequence[j+1]]
		if len(ProteinSequence)-i==0:
			Result['MoreauBrotoAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)),3)
		else:
			Result['MoreauBrotoAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)-i),3)

	return Result


def CalculateEachMoranAuto(ProteinSequence,AAP,AAPName):
	"""
	you can use the function to compute MoranAuto descriptors for different properties based on AADs.
	Input: protein is a pure protein sequence.
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	Output: result is a dict form containing 30 Moran autocorrelation descriptors based on the given property.
	"""

	AAPdic=NormalizeEachAAP(AAP)

	cds=0
	for i in AALetter:
		cds=cds+(ProteinSequence.count(i))*(AAPdic[i])
	Pmean=cds/len(ProteinSequence)

	cc=[]
	for i in ProteinSequence:
		cc.append(AAPdic[i])

	K=(_std(cc,ddof=0))**2

	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
				
			temp=temp+(AAPdic[ProteinSequence[j]]-Pmean)*(AAPdic[ProteinSequence[j+i]]-Pmean)
		if len(ProteinSequence)-i==0:
			Result['MoranAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence))/K,3)
		else:
			Result['MoranAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)-i)/K,3)

	return Result


def CalculateEachGearyAuto(ProteinSequence,AAP,AAPName):

	"""
	you can use the function to compute GearyAuto descriptors for different properties based on AADs.
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	Output: result is a dict form containing 30 Geary autocorrelation descriptors based on the given property.
	"""

	AAPdic=NormalizeEachAAP(AAP)

	cc=[]
	for i in ProteinSequence:
		cc.append(AAPdic[i])

	K=((_std(cc))**2)*len(ProteinSequence)/(len(ProteinSequence)-1)
	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
				
			temp=temp+(AAPdic[ProteinSequence[j]]-AAPdic[ProteinSequence[j+i]])**2
		if len(ProteinSequence)-i==0:
			Result['GearyAuto'+AAPName+str(i)]=round(temp/(2*(len(ProteinSequence)))/K,3)
		else:
			Result['GearyAuto'+AAPName+str(i)]=round(temp/(2*(len(ProteinSequence)-i))/K,3)
	return Result


##################################################################################################

def CalculateNormalizedMoreauBrotoAuto(ProteinSequence,AAProperty,AAPropertyName): 

	"""
	A method used for computing MoreauBrotoAuto for all properties.
	Output: result is a dict form containing 30*p Normalized Moreau-Broto autocorrelation descriptors based on the given properties.
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])
	return Result


def CalculateMoranAuto(ProteinSequence,AAProperty,AAPropertyName):  
	"""
	A method used for computing MoranAuto for all properties 	
	Output: result is a dict form containing 30*p Moran autocorrelation descriptors based on the given properties.
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachMoranAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])

	return Result



def CalculateGearyAuto(ProteinSequence,AAProperty,AAPropertyName):  
	"""
	A method used for computing GearyAuto for all properties 
	Output: result is a dict form containing 30*p Geary autocorrelation descriptors based on the given properties.
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachGearyAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])

	return Result


########################NormalizedMoreauBorto##################################
def CalculateNormalizedMoreauBrotoAutoHydrophobicity(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on hydrophobicity.	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on Hydrophobicity.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result


def CalculateNormalizedMoreauBrotoAutoAvFlexibility(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on AvFlexibility.
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on AvFlexibility.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateNormalizedMoreauBrotoAutoPolarizability(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on Polarizability.
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on Polarizability.
	"""
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result


def CalculateNormalizedMoreauBrotoAutoFreeEnergy(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on FreeEnergy.descriptors based on FreeEnergy.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result



def CalculateNormalizedMoreauBrotoAutoResidueASA(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on ResidueASA.	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on ResidueASA.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result


def CalculateNormalizedMoreauBrotoAutoResidueVol(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on ResidueVol.
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on ResidueVol.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
	
def CalculateNormalizedMoreauBrotoAutoSteric(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on Steric.
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on Steric.
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Steric,'_Steric')
	return result


def CalculateNormalizedMoreauBrotoAutoMutability(ProteinSequence):

	"""
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on Mutability.
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation descriptors based on Mutability.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Mutability,'_Mutability')
	return result


def CalculateMoranAutoHydrophobicity(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on hydrophobicity.
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on hydrophobicity.
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	

def CalculateMoranAutoAvFlexibility(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on AvFlexibility
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on AvFlexibility.
	"""	
	result=CalculateEachMoranAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateMoranAutoPolarizability(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on Polarizability.	
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on Polarizability.
	"""
	result=CalculateEachMoranAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result


def CalculateMoranAutoFreeEnergy(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on FreeEnergy.
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on FreeEnergy.
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result



def CalculateMoranAutoResidueASA(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on ResidueASA.	
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on ResidueASA.
	"""
	result=CalculateEachMoranAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result


def CalculateMoranAutoResidueVol(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on ResidueVol.
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on ResidueVol.
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
	
def CalculateMoranAutoSteric(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on AutoSteric.
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on AutoSteric.
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Steric,'_Steric')
	return result


def CalculateMoranAutoMutability(ProteinSequence):

	"""
	Calculte the MoranAuto Autocorrelation descriptors based on Mutability.
	Output: result is a dict form containing 30 Moran Autocorrelation descriptors based on Mutability.
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Mutability,'_Mutability')
	return result




################################GearyAuto#####################################
def CalculateGearyAutoHydrophobicity(ProteinSequence):

	"""
	Calculte the GearyAuto Autocorrelation descriptors based on hydrophobicity.
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on hydrophobicity.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	

def CalculateGearyAutoAvFlexibility(ProteinSequence):

	"""
	Calculte the GearyAuto Autocorrelation descriptors based on AvFlexibility.	
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on AvFlexibility.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateGearyAutoPolarizability(ProteinSequence):

	"""
	Calculte the GearyAuto Autocorrelation descriptors based on Polarizability.	
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on Polarizability.
    """
	
	result=CalculateEachGearyAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result

##############################################################################################
def CalculateGearyAutoFreeEnergy(ProteinSequence):
	
	"""
	Calculte the GearyAuto Autocorrelation descriptors based on FreeEnergy.
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on FreeEnergy.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result


##############################################################################################
def CalculateGearyAutoResidueASA(ProteinSequence):
	
	"""
	Calculte the GearyAuto Autocorrelation descriptors based on ResidueASA.	
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on ResidueASA.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result

##############################################################################################
def CalculateGearyAutoResidueVol(ProteinSequence):
	
	"""
	Calculte the GearyAuto Autocorrelation descriptors based on ResidueVol.
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on ResidueVol.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
################################################################################################	
def CalculateGearyAutoSteric(ProteinSequence):
	"""
	Calculte the GearyAuto Autocorrelation descriptors based on Steric.
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on Steric.
	"""
	result=CalculateEachGearyAuto(ProteinSequence,_Steric,'_Steric')
	return result

################################################################################################

def CalculateGearyAutoMutability(ProteinSequence):
	"""
	Calculte the GearyAuto Autocorrelation descriptors based on Mutability.	
	Output: result is a dict form containing 30 Geary Autocorrelation descriptors based on Mutability.
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Mutability,'_Mutability')
	return result
##################################################################################################

def CalculateNormalizedMoreauBrotoAutoTotal(ProteinSequence):
	"""
	A method used for computing normalized Moreau Broto autocorrelation descriptors based on 8 proterties of AADs.
	Output: result is a dict form containing 30*8=240 normalized Moreau Broto autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(CalculateNormalizedMoreauBrotoAutoHydrophobicity(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoAvFlexibility(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoPolarizability(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoFreeEnergy(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoResidueASA(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoResidueVol(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoSteric(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoMutability(ProteinSequence))
	return result
#################################################################################################
def CalculateMoranAutoTotal(ProteinSequence):
	"""
	A method used for computing Moran autocorrelation descriptors based on 8 properties of AADs.
	Output: result is a dict form containing 30*8=240 Moran
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(CalculateMoranAutoHydrophobicity(ProteinSequence))
	result.update(CalculateMoranAutoAvFlexibility(ProteinSequence))
	result.update(CalculateMoranAutoPolarizability(ProteinSequence))
	result.update(CalculateMoranAutoFreeEnergy(ProteinSequence))
	result.update(CalculateMoranAutoResidueASA(ProteinSequence))
	result.update(CalculateMoranAutoResidueVol(ProteinSequence))
	result.update(CalculateMoranAutoSteric(ProteinSequence))
	result.update(CalculateMoranAutoMutability(ProteinSequence))
	return result
##################################################################################################
def CalculateGearyAutoTotal(ProteinSequence):
	"""
	A method used for computing Geary autocorrelation descriptors based on 8 properties of AADs.	
	Output: result is a dict form containing 30*8=240 Geary
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(CalculateGearyAutoHydrophobicity(ProteinSequence))
	result.update(CalculateGearyAutoAvFlexibility(ProteinSequence))
	result.update(CalculateGearyAutoPolarizability(ProteinSequence))
	result.update(CalculateGearyAutoFreeEnergy(ProteinSequence))
	result.update(CalculateGearyAutoResidueASA(ProteinSequence))
	result.update(CalculateGearyAutoResidueVol(ProteinSequence))
	result.update(CalculateGearyAutoSteric(ProteinSequence))
	result.update(CalculateGearyAutoMutability(ProteinSequence))
	return result

##################################################################################################
def CalculateAutoTotal(ProteinSequence):
	"""	
	Output: result is a dict form containing 30*8*3=720 normalized Moreau Broto, Moran, and Geary	
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(CalculateNormalizedMoreauBrotoAutoTotal(ProteinSequence))
	result.update(CalculateMoranAutoTotal(ProteinSequence))
	result.update(CalculateGearyAutoTotal(ProteinSequence))
	return result








#########################################AAC####################################################
def CalculateAAComposition(ProteinSequence):

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round((float(ProteinSequence.count(i)) / LengthSequence)*100, 2)
    return Result

#20 features


#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):

    #400 dipepdite features

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round((float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1))*100, 2)
    return Result



#############################################################################################

def Getkmers():

    #8000 tripeptide name
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers



#############################################################################################
def GetSpectrumDict(proteinsequence):
  #8000 tripeptide features
     
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence)*100)
    return result



#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):

    #8420 all uni,bi,tri features

    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))

    result.update(GetSpectrumDict(ProteinSequence))

    return result



#############################################################################################
def secondary_structure_fraction(p): 

    aa_percentages = CalculateAAComposition(p) 
    helix = sum(aa_percentages[r] for r in 'VIYFWL') 
    turn = sum(aa_percentages[r] for r in 'NPGS') 
    sheet = sum(aa_percentages[r] for r in 'EMAL') 
    return helix, turn, sheet

def CalculateAACompositioncount(ProteinSequence):
    


   
    Result = {}
    for i in AALetter:
        Result[i] = (float(ProteinSequence.count(i)))
    return Result

def isoelectric_point(p): 
          aa_content =CalculateAACompositioncount(p) 
   
          ie_point = IsoelectricPoint.IsoelectricPoint(p, aa_content) 
          return ie_point.pi()    
#############################################################################################    
#start feature value storing
#fbr=open('binding.txt','r')
#ff=open('ready_binding.txt','w')
fflbl=open('lbl.txt','r')

def ready_file():
 
 for line in fflbl:
     
    c=''
    a=line
    splitted = a.split('\t')
    first = splitted[0]

    f=''
    fs = line[len(first)+1:(len(line)-2)]
    fs=list(fs.split(';'))

    for i in fs:
       if i!='':         
         with open('binding.txt','r') as fbr:
            for bind in fbr:
             if i in bind:
                aa=line
                splt = aa.split()
                f = splt[0]
                c=c+i+';'

                if f!='':
                 with open('ready_binding_all_dataset_updated.txt', 'a') as ff: 
                    ff.write(f+' '+i+';'+'\n')
'''
    if f!='':
     with open('ready_super_family.txt', 'a') as f: 
      f.write(first+' '+c+'\n')
'''      
     #print('find---',f,' ',c)
             #else:
                 #print('no binding site')
    #print('1 lbl checked')
         
                 
    
    #print('ok')             
    #print(dlist)
     #dlist[:] = []
     
          
def extract_feature_for_prediction():

     
     for record in SeqIO.parse ( "zebrafish_sequence.fasta", "fasta" ):
         #record=list(SeqIO.parse ( "sequence.fasta", "fasta" ))
         if 'O13033' in record.description:
            
            p=str(record.seq)
            CalculateAAComposition(p)
            CalculateDipeptideComposition(p)
            GetSpectrumDict(p)
            res = CalculateAADipeptideComposition(p )
            Autocorellation=CalculateAutoTotal(protein)
            Ctried=CalculateConjointTriad(protein)
            CTD=CalculateCTD(protein)
            SCN=GetSequenceOrderCouplingNumberTotal(protein,maxlag=30)
            QSO=GetQuasiSequenceOrderp(protein,maxlag=30,distancematrix=_Distance1)
            
            
          
            flist=list(res.values())+list(Autocorellation.values())+list(Ctried.values())+list(CTD.values())+list(SCN.values())+list(QSO.values())
           
            
            
            r=list(secondary_structure_fraction(p))
            flist.append(r[0])
            flist.append(r[1])
            flist.append(r[2])
            flist.append(isoelectric_point(p))
            a=np.array(flist)
          

     return (a.reshape(1,-1))
                 
                 
                 
fflbl=open('ready_binding_all_dataset.txt','r') 

def store_feature_value():

 for line in fflbl:
     
     a=line
     splitted = a.split()
     first = splitted[0]
     
     for record in SeqIO.parse ( "roundworm sequence.fasta", "fasta" ):
         #record=list(SeqIO.parse ( "sequence.fasta", "fasta" ))
         if line[:len(first)] in record.description:
            
            p=str(record.seq)
            CalculateAAComposition(p)
            CalculateDipeptideComposition(p)
            GetSpectrumDict(p)
            res = CalculateAADipeptideComposition(p )
            Autocorellation=CalculateAutoTotal(protein)
            Ctried=CalculateConjointTriad(protein)
            CTD=CalculateCTD(protein)
            SCN=GetSequenceOrderCouplingNumberTotal(protein,maxlag=30)
            QSO=GetQuasiSequenceOrderp(protein,maxlag=30,distancematrix=_Distance1)
            
            
          
            flist=list(res.values())+list(Autocorellation.values())+list(Ctried.values())+list(CTD.values())+list(SCN.values())+list(QSO.values())
           
            
            
            r=list(secondary_structure_fraction(p))
            flist.append(r[0])
            flist.append(r[1])
            flist.append(r[2])
            flist.append(isoelectric_point(p))
          
            #fs = line[len(first)+1:(len(line)-2)].split(';')
            
            fs = line[len(first)+1:(len(line)-2)]
            flist.append(fs)
            flist.append(line[:len(first)])
            
            myFile = open('binding_feature_all_dataset.csv', 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(flist)

###################################################################################################
        
#store features name 
def store_feature_name():

 CalculateAAComposition(protein)
 CalculateDipeptideComposition(protein)

 a=[]    
 GetSpectrumDict(protein)

 AAC = CalculateAADipeptideComposition(protein)
 
 Autocorellation=CalculateAutoTotal(protein)
 
 Ctried=CalculateConjointTriad(protein)
 
 CTD=CalculateCTD(protein)
 
 SCN=GetSequenceOrderCouplingNumberTotal(protein,maxlag=30)
 
 QSO=GetQuasiSequenceOrderp(protein,maxlag=30,distancematrix=_Distance1)




 b=list(AAC.keys())
 c=list(Autocorellation.keys())
 d=list(Ctried.keys())
 e=list(CTD.keys())
 f=list(SCN.keys())
 g=list(QSO.keys())
 
 a=b+c+d+e+f+g
 a.append('helix')
 a.append('turn')
 a.append('sheet')
 a.append('iso')
 
 
 a.append('labels')
 '''
 un=list()
 albl=open('ready_binding_zebra','r')
 for aline in albl:
     sv=aline
     splt=sv.split()
     un.append(splt[1])
     
 un=list(OrderedDict.fromkeys(un))
 
 a=a+un
 '''
 a.append('pid')
 myFile = open('new_BY_domain_feature_seprate.csv', 'a')
 with myFile:
  writer = csv.writer(myFile)
  writer.writerow(a)

###################################################################################################
# end store feature name 



if __name__ == "__main__":
    
 protein = 'MNTDQQPYQGQTDYTQGPGNGQSQEQDYDQYGQPLYPSQADGYYDPNVAAGTEADMYGQQ'
 print('processing: ') 
 
 #ready_file()
 #store_feature_name()
 #store_feature_value()
 


 import pandas as pd
 df = pd.read_csv('binding_feature_all_dataset.csv')
 #print(df)
 data = df.drop('pid', axis=1)
 data = data.drop('labels', axis=1)
# data=data.drop(data.iloc[:, 9913:9957], axis=1) #9913 included and 9958-1 included 
 
 #target_labels = df.iloc[:,9913:9914]
 
 
 
 
 target_labels=df['labels']
 
 unique_target_labels=df['labels'].unique().tolist()
 print('unique labels',len(unique_target_labels))
 #print(extract_feature_for_prediction())
 

'''
IPR017441;
IPR018247;
IPR020583;
IPR013838;
'''

'''
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

target_labels=MultiLabelBinarizer().fit_transform(target_labels)
data_train, data_test, target_labels_train, target_labels_test = train_test_split(data, target_labels,test_size=0.3)

#classifier = LabelPowerset(GaussianNB())
classifier = BinaryRelevance(GaussianNB())


classifier.fit(data_train, target_labels_train)


predictions = classifier.predict(data_test)


print('Domain predictions accuracy: ',accuracy_score(target_labels_test,predictions)*100)

'''



#target_labels=MultiLabelBinarizer().fit_transform(target_labels)

'''
from sklearn.preprocessing import label_binarize
target_labels = label_binarize(target_labels,target_labels)#[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42])
'''
#total_target_labels = target_labels.shape[1]

data_train, data_test, target_labels_train, target_labels_test = train_test_split(data, target_labels,random_state=0,test_size=0.3)

random_forest = RandomForestClassifier(n_jobs=2,n_estimators=600, random_state=0,max_features='auto',criterion='entropy',min_samples_leaf=2,oob_score=True)

random_forest.fit(data_train,target_labels_train ) #y train_y.values.ravel())

scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

#result=cross_val_score(random_forest,X_train,y_train,cv=10,scoring=scoring)
#print('result ',result)



y_predict = random_forest.predict(data_test)
#print('for prediction give data this form',X_test)
y2_predict = random_forest.predict(data_test)

print('precission: ',precision_score(target_labels_test, y_predict, average='micro'))

print('Recall: ',recall_score(target_labels_test,y_predict,average='micro'))

print('New predcition Accuracy (test): ',(accuracy_score(target_labels_test, y_predict)*100))


training_predict = random_forest.predict(data_train)
print('Training Accuracy is:',(accuracy_score(target_labels_train, training_predict)*100))


print('F1 score: ',f1_score(target_labels_test, y_predict, average='micro'))

import pickle
modelname = 'binding_site_treained_model.sav'
pickle.dump(random_forest, open(modelname, 'wb'))
 

loaded_model = pickle.load(open(modelname, 'rb'))


'''
result = loaded_model.score(X_test, Y_test)
print(result)
'''

#Xnew = [[...], [...]]

#pidd=str(input("protein id for prediction :"))



ynew = loaded_model.predict(extract_feature_for_prediction())
print("your binding site against this O13033 protein id is ",ynew)


''''
tick_marks=np.arange(len(target_labels_test))
plt.xticks(tick_marks)


plt.show(confusion_matrix(target_labels_test, y_predict))
#print ('Confusion matrix ', (confusion_matrix(target_labels_test, y_predict)))
'''
###########################################Plot ROC curve############################################
'''
from sklearn.metrics import roc_curve, auc
#from scipy import interp
#from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()

total_target_labels = target_labels.shape[1]

for i in range(total_target_labels):
    fpr[i], tpr[i], _ = roc_curve(target_labels_test[:, i], y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(target_labels_test.ravel(), y_predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()

lw = 2
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

plt.show()

plt.savefig("rocBinding.pdf")
'''
####################################################################################################
'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled = df(scaler.transform(X_train))
X_test_scaled = df(scaler.transform(X_test))

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
cpts = df(pca.transform(X_train))
x_axis = np.arange(1, pca.n_components_+1)
pca_scaled = PCA()
pca_scaled.fit(X_train_scaled)
cpts_scaled = df(pca.transform(X_train_scaled))
'''




#print(df.head())
#print(X)
#print(y)


#y = pd.factorize(df['labels'])[0]
#print(y)

######################################SVM#################################################
'''
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


data_train, data_test, target_labels_train, target_labels_test = train_test_split(data, target_labels,test_size=0.3)
 
# initialize classifier
# you can set hyperparameters here. (use documentation)
clf = SVC()
 
# train the classifier on the training data
clf.fit(data_train, target_labels_train) 
SVC(C=1.0, cache_size=600, class_weight=None, degree=3,gamma='auto',kernel='rbf',shrinking=True,verbose=False,random_state=None)

y_predict = clf.predict(data_test)
print('Your svm predcition Accuracy is:',(accuracy_score(target_labels_test, y_predict)*100))

'''


#####################################SVC####################################################

'''
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
y=MultiLabelBinarizer().fit_transform(y)

random_state = np.random.RandomState(10)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=random_state)
 
#our pipeline transforms our text into a vector and then applies OneVsRest using LinearSVC
 
pipeline = Pipeline([
 

('clf', OneVsRestClassifier(LinearSVC()))])
 



print('Model is training......')

pipeline.fit(X_train,y_train)


y_predict = pipeline.predict(X_test)


print('Your new predcition Accuracy is:',(accuracy_score(y_test, y_predict)*100))

x_predict = pipeline.predict(X_train)

print('Your training Accuracy is:',(accuracy_score(y_train, x_predict)*100))
'''


###############################################RF####################################################

'''

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

random_forest = RandomForestClassifier(n_jobs=2,n_estimators=600, random_state=0,max_features='auto',criterion='entropy',min_samples_leaf=2,oob_score=True)
#result=cross_val_score(random_forest,X_train,y_train,cv=10,scoring=scoring)

random_forest.fit(X_train,y_train ) #y

y_predict = random_forest.predict(X_test)
#print('for prediction give data this form',X_test)
y2_predict = random_forest.predict(X_test)

print('precission: ',precision_score(y_test, y_predict, average='micro'))
print('Recall: ',recall_score(y_test,y_predict,average='micro'))
print('New predcition Accuracy: ',(accuracy_score(y_test, y_predict)))

x_predict = random_forest.predict(X_train)

#print('Training Accuracy is:',(accuracy_score(y_train, x_predict)*100))
print('F1 score: ',f1_score(y_test, y_predict, average='micro'))
'''

#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.values.argmax(axis=1), y2_predict.argmax(axis=1))

#print ("Confusion matrix: ", confusion_matrix(y_test.values.argmax(axis=1), y2_predict.argmax(axis=1) ))

#print('Area under curve: ',auc(false_positive_rate, true_positive_rate))




'''
# using Label Powerset
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
y=MultiLabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.03)
# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)


print('predictions accuracy: ',accuracy_score(y_test,predictions)*100)
'''



#print ('Confusion matrix ', (confusion_matrix(y_test, x_predict)))

#from sklearn.metrics import confusion_matrix
#cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
#sns.heatmap(cm, annot=True)


#This is simply a matrix whose diagonal values are true positive counts,
#while off-diagonal values are false positive and false negative counts for each class against the other

'''
###################################################################################################

# using binary relevance
#from skmultilearn.problem_transform import BinaryRelevance
#from sklearn.naive_bayes import GaussianNB
#from skmultilearn.problem_transform import LabelPowerset
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())
#classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)
print(accuracy_score(y_test,predictions))


##################################################################################################

from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=300)

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

accuracy_score(y_test,predictions)
print('knn',accuracy_score(y_test,predictions)*100)


from pandas import read_csv
from sklearn.decomposition import PCA
# load data
names=[]
dataframe = read_csv('feature.csv', names=res.keys())
array = dataframe.values
X = array[:,0:419]
Y = array[:,419]
# feature extraction
pca = PCA(n_components=25)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)

                                 
#f_classif  ' '.join('{0}{1}'.format(key, val) for key, val in sorted(res.items())))
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
ftr=vec.fit_transform(res)

    
        #  print('features:',ftr)

#print('features name')
#print(vec.get_feature_names())


#f.append['labels']

import pandas as pd 
df = pd.DataFrame(vec.get_feature_names())
df.to_csv("ftr.csv")

ff=open('feature.txt','a')
for i in vec.get_feature_names():
    ff.write(i+'\t')
    

ff.close()    
array = ftr
X = array[:,0:8419]
Y = array[:,8419]
    # feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
    # summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
    # summarize selected features
print('selected feature',features[0:20,:]) 

  
AAC = CalculateAAComposition(p)
        
         #print (AAC)
DIP = CalculateDipeptideComposition(p)
         #print ('Dipeptide: ',DIP)
         #print('Dipeptide total features:',len(DIP))
         #your_dict.keys()
         #your_dict.values()

    
spectrum = GetSpectrumDict(p)
         #print ('spectrum: ',spectrum)
res = CalculateAADipeptideComposition(p )
import csv 
myFile = open('feature.csv', 'a')
with myFile:
 writer = csv.writer(myFile)
 writer.writerow(res.keys())
'''

print("successfuly: ")
#fbr.close()  
#ff.close()   
fflbl.close()
#fflabel.close()

 
 

    
    
 

