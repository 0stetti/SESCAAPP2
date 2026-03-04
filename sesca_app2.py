#!/usr/bin/env python3
"""
SESCA Web App - Predicao de Espectros de Dicroismo Circular (CD)
Implementacao pure-Python do algoritmo SESCA (Nagy et al., JCTC 2019).
Executa inteiramente no Streamlit Cloud sem dependencias externas.

Algoritmo:
  1. Parse PDB -> extrair atomos do backbone (N, CA, C)
  2. Calcular angulos diedros phi/psi
  3. Classificar estrutura secundaria via DISICL (dihedral-based)
  4. Calcular fracoes de cada classe SS
  5. Combinar espectros-base ponderados pelas fracoes -> espectro CD

Referencia:
  Nagy, G. et al. SESCA: Predicting Circular Dichroism Spectra from Protein
  Molecular Structure. J. Chem. Theory Comput. 15, 5087-5102 (2019).
  doi: 10.1021/acs.jctc.9b00203

  Nagy, G. & Oostenbrink, C. DISICL: Dihedral-based Segment Identification
  and Classification. J. Chem. Inf. Model. 54, 266-277 (2014).
  doi: 10.1021/ci400541d

Executar com: streamlit run sesca_app2.py
"""

import io
import csv
import math
import tempfile
import urllib.request
from pathlib import Path
from collections import OrderedDict

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# =============================================================================
#  DADOS DOS CONJUNTOS DE BASE EMBUTIDOS
# =============================================================================
# Extraidos diretamente dos arquivos .dat oficiais do SESCA v0.97

# --- DS-dT basis set (recomendado para DISICL detailed) ---
# Combination matrix: Alpha = 3H+ALH+PIH, Beta = EBS+NBS, Coil = tudo mais
# 19 classes DISICL -> 3 espectros-base
DS_DT_COMBINATION = {
    "3H": "Alpha", "ALH": "Alpha", "PIH": "Alpha",
    "EBS": "Beta", "NBS": "Beta",
    "PP": "Coil", "TI": "Coil", "TII": "Coil", "TVIII": "Coil",
    "GXT": "Coil", "SCH": "Coil", "HP": "Coil", "TC": "Coil",
    "HC": "Coil", "BC": "Coil", "BU": "Coil", "LTII": "Coil",
    "LHH": "Coil", "UC": "Coil",
}

DS_DT_SPECTRA = {
    "basis_names": ["Alpha", "Beta", "Coil"],
    "wavelengths": [
        269,268,267,266,265,264,263,262,261,260,259,258,257,256,255,254,253,252,251,250,
        249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,
        229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,
        209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,
        189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,
    ],
    "Alpha": [
        0.162,0.301,0.051,-0.068,-0.094,-0.037,-0.003,-0.114,-0.238,-0.219,-0.078,-0.004,
        -0.023,-0.075,-0.170,-0.170,-0.148,-0.241,-0.320,-0.333,-0.373,-0.411,-0.506,
        -0.734,-1.039,-1.356,-1.769,-2.396,-3.106,-4.033,-5.269,-6.631,-8.194,-9.926,
        -11.756,-13.829,-16.110,-18.598,-21.210,-23.870,-26.401,-28.751,-30.776,-32.526,
        -33.877,-34.807,-35.347,-35.549,-35.399,-35.021,-34.464,-33.736,-33.112,-32.608,
        -32.170,-31.936,-31.862,-31.893,-32.178,-32.402,-32.132,-30.864,-28.084,-23.519,
        -17.508,-10.195,-1.956,7.005,16.798,27.445,38.629,49.679,59.526,67.262,72.467,
        75.213,75.757,74.358,71.404,67.071,61.496,55.454,49.353,43.438,38.258,33.707,
        29.659,26.390,23.771,21.542,19.787,18.236,16.662,14.988,13.130,
    ],
    "Beta": [
        0.408,0.394,0.189,0.099,0.054,0.059,0.165,0.125,0.034,-0.102,-0.150,-0.210,
        -0.173,-0.135,-0.121,-0.128,-0.104,-0.110,-0.172,-0.140,-0.050,0.064,0.229,
        0.426,0.601,0.715,0.841,0.889,1.018,1.209,1.478,1.715,2.047,2.414,2.977,
        3.509,4.022,4.281,4.244,3.996,3.544,2.883,2.033,1.007,-0.134,-1.389,-2.725,
        -3.987,-5.098,-5.935,-6.653,-7.218,-7.443,-7.348,-6.712,-5.577,-4.063,-1.979,
        0.782,3.922,7.790,12.415,17.629,23.657,29.916,35.539,40.538,44.444,47.191,
        48.982,50.044,50.022,49.083,46.939,43.323,38.422,32.551,26.154,19.721,13.316,
        6.984,1.052,-4.903,-10.286,-14.801,-18.355,-20.867,-23.132,-25.114,-26.835,
        -28.055,-28.459,-28.562,-28.593,-28.304,
    ],
    "Coil": [
        -0.149,-0.185,-0.048,0.008,0.023,0.001,-0.050,-0.001,0.076,0.143,0.121,0.109,
        0.074,0.057,0.074,0.094,0.084,0.093,0.121,0.078,0.022,-0.054,-0.136,-0.199,
        -0.272,-0.331,-0.384,-0.396,-0.426,-0.431,-0.463,-0.479,-0.532,-0.610,-0.801,
        -0.968,-1.191,-1.367,-1.467,-1.528,-1.565,-1.576,-1.621,-1.668,-1.745,-1.836,
        -1.937,-2.058,-2.245,-2.492,-2.757,-3.016,-3.268,-3.457,-3.737,-4.035,-4.441,
        -5.000,-5.741,-6.608,-7.771,-9.195,-10.926,-13.044,-15.264,-17.323,-19.102,
        -20.406,-21.222,-21.712,-21.899,-21.724,-21.241,-20.268,-18.701,-16.596,-14.095,
        -11.356,-8.748,-6.284,-4.036,-2.180,-0.654,0.461,1.017,0.984,0.433,-0.327,
        -1.216,-2.074,-3.036,-4.168,-5.172,-5.950,-6.534,
    ],
}

# --- DS5-4 basis set (recomendado para estruturas com loops) ---
DS5_4_COMBINATION = {
    "3H": "Helix2", "ALH": "Helix1", "PIH": "Turn2",
    "EBS": "Beta1", "NBS": "Beta1",
    "PP": "Turn1", "TI": "Turn1", "TII": "Helix2", "TVIII": "Helix1",
    "GXT": "Turn1", "SCH": "Turn2", "HP": "Turn2", "TC": "Other",
    "HC": "Turn2", "BC": "Turn2", "BU": "Turn1", "LTII": "Turn2",
    "LHH": "Turn2", "UC": "Other",
}

DS5_4_SPECTRA = {
    "basis_names": ["Helix1", "Beta1", "Helix2", "Turn1", "Turn2", "Other"],
    "wavelengths": [
        269,268,267,266,265,264,263,262,261,260,259,258,257,256,255,254,253,252,251,250,
        249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,
        229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,
        209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,
        189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,
    ],
    "Helix1": [
        0.201,0.507,0.090,-0.167,-0.183,-0.093,-0.009,-0.184,-0.406,-0.367,-0.029,0.108,
        0.107,0.010,-0.253,-0.281,-0.211,-0.316,-0.376,-0.326,-0.342,-0.284,-0.387,
        -0.717,-1.087,-1.408,-1.845,-2.425,-3.087,-4.079,-5.415,-6.885,-8.684,-10.612,
        -12.623,-14.904,-17.414,-20.223,-23.226,-26.303,-29.222,-31.943,-34.214,-36.180,
        -37.670,-38.604,-39.118,-39.324,-39.066,-38.566,-37.943,-37.020,-36.364,-35.862,
        -35.444,-35.266,-35.400,-35.498,-36.093,-36.637,-36.639,-35.641,-33.249,-29.005,
        -23.458,-16.530,-8.350,0.766,11.198,22.996,35.599,48.344,60.015,69.578,76.820,
        81.687,83.968,83.945,81.944,78.082,72.670,66.546,59.989,53.338,47.449,42.168,
        37.428,33.412,30.133,27.161,24.769,22.762,20.702,18.657,16.268,
    ],
    "Beta1": [
        0.445,0.209,0.141,0.173,0.132,0.131,0.199,0.200,0.209,0.076,-0.163,-0.292,
        -0.291,-0.186,-0.005,0.034,-0.006,-0.045,-0.155,-0.152,-0.071,-0.026,0.177,
        0.497,0.773,0.962,1.164,1.174,1.304,1.558,1.942,2.341,2.945,3.511,4.297,
        5.012,5.752,6.362,6.790,7.097,7.225,7.137,6.714,6.033,5.235,4.212,3.044,
        1.955,0.850,-0.067,-0.789,-1.485,-1.691,-1.523,-0.805,0.368,2.052,4.142,
        7.134,10.477,14.402,18.935,24.032,29.800,35.865,41.264,45.853,49.338,51.252,
        51.795,51.366,49.670,47.025,43.179,37.598,30.584,23.005,15.126,7.647,0.612,
        -6.005,-11.805,-17.175,-21.707,-25.305,-27.930,-29.553,-30.830,-32.028,-32.838,
        -33.187,-32.966,-32.467,-31.999,-31.104,
    ],
    "Helix2": [
        -0.287,-2.908,-0.704,1.236,1.313,1.041,0.538,1.189,2.480,2.252,-0.582,-1.383,
        -1.643,-0.958,1.307,1.783,1.123,1.071,0.653,-0.115,-0.184,-1.511,-1.378,
        -0.008,0.926,0.915,1.478,0.543,-0.380,0.123,0.906,1.415,3.998,5.128,5.875,
        6.443,6.920,8.356,11.495,14.641,18.036,21.749,23.945,25.610,27.073,26.888,
        26.731,27.711,27.444,26.821,27.066,25.607,26.489,28.270,29.778,30.997,33.795,
        34.458,37.472,40.041,40.748,41.173,43.230,45.068,49.361,54.257,57.716,62.663,
        64.524,62.969,59.793,53.680,45.475,34.799,17.876,-3.415,-23.127,-41.908,-57.673,
        -70.105,-79.494,-85.526,-86.773,-85.133,-82.626,-79.625,-76.476,-71.668,-67.955,
        -62.488,-56.784,-53.297,-49.333,-44.755,-40.080,
    ],
    "Turn1": [
        0.806,1.234,0.258,-0.716,-0.907,-0.561,-0.463,-0.522,-0.759,-0.511,0.606,0.677,
        0.401,0.536,0.029,0.021,-0.069,-0.433,-0.583,-0.052,-0.460,-0.012,-0.020,
        -0.395,-0.807,-0.156,-0.783,-1.141,-1.279,-2.553,-3.785,-3.903,-5.869,-6.698,
        -6.961,-7.645,-7.563,-6.968,-7.358,-5.919,-4.611,-3.753,-2.625,-1.506,0.058,
        2.767,4.881,5.637,5.940,5.798,5.805,6.206,5.403,3.603,2.253,0.613,-1.193,
        -3.288,-4.943,-6.066,-5.612,-5.464,-5.446,-5.347,-5.102,-4.646,-5.352,-9.219,
        -15.600,-24.543,-34.800,-44.366,-53.085,-59.692,-63.524,-65.813,-64.921,-63.167,
        -59.721,-54.870,-50.446,-44.671,-37.669,-29.716,-21.454,-13.417,-6.506,0.815,
        7.370,13.964,19.210,23.533,26.635,25.900,28.292,
    ],
    "Turn2": [
        -0.183,0.105,-0.075,-0.364,-0.173,-0.027,0.211,0.096,-0.087,-0.044,0.362,0.560,
        0.618,0.566,0.096,0.028,0.187,0.211,0.326,0.541,0.711,0.868,0.726,0.422,
        0.290,0.268,0.450,0.802,1.072,1.189,1.267,1.086,0.774,0.170,-0.694,-1.649,
        -2.824,-4.382,-5.532,-6.982,-8.064,-8.838,-9.250,-9.746,-9.866,-9.753,-9.441,
        -8.982,-8.149,-7.434,-7.093,-6.464,-6.251,-5.844,-5.924,-6.397,-7.677,-8.815,
        -11.322,-14.340,-18.451,-23.419,-29.878,-37.317,-45.779,-53.720,-59.292,-61.663,
        -60.310,-55.410,-48.264,-39.361,-29.198,-18.465,-6.636,5.684,16.290,24.636,
        30.725,33.988,36.051,37.116,36.583,34.030,31.283,27.534,23.395,19.004,14.601,
        10.014,6.286,3.141,0.080,-0.880,-3.192,
    ],
    "Other": [
        -0.467,-0.381,0.008,0.358,0.275,0.015,-0.243,-0.136,0.010,0.063,-0.177,-0.224,
        -0.187,-0.390,-0.222,-0.224,-0.202,-0.034,0.084,-0.268,-0.387,-0.624,-0.751,
        -0.812,-0.962,-1.414,-1.651,-1.741,-1.928,-1.772,-1.766,-1.919,-1.773,-1.577,
        -1.602,-1.285,-1.225,-1.168,-1.289,-1.813,-2.612,-3.486,-4.490,-5.242,-6.513,
        -8.080,-9.545,-10.827,-12.074,-13.041,-13.952,-14.887,-15.469,-15.901,-16.253,
        -16.118,-15.809,-15.440,-14.984,-14.469,-13.812,-12.629,-10.876,-8.899,-6.594,
        -4.429,-2.868,-2.057,-1.635,-1.762,-2.210,-3.066,-4.402,-5.678,-6.611,-6.712,
        -6.278,-4.073,-1.717,1.360,4.328,5.884,6.559,7.091,6.249,4.975,3.302,0.952,
        -1.010,-3.093,-5.654,-8.235,-10.196,-12.093,-13.596,
    ],
}

# --- DSSP-1 basis set (aproximacao por phi/psi) ---
DSSP_1_COMBINATION_MAP = {
    "Helix": ["H"],
    "Beta": ["E"],
    "Bridge": ["B"],
    "Other": ["T", "S", "C"],
}

DSSP_1_SPECTRA = {
    "basis_names": ["Helix", "Beta", "Bridge", "Other"],
    "wavelengths": [
        269,268,267,266,265,264,263,262,261,260,259,258,257,256,255,254,253,252,251,250,
        249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,
        229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,
        209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,
        189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,
    ],
    "Helix": [
        0.257,0.349,0.057,-0.121,-0.112,-0.037,0.026,-0.068,-0.181,-0.172,-0.031,0.056,
        0.054,0.067,-0.013,-0.032,-0.084,-0.209,-0.326,-0.268,-0.278,-0.238,-0.319,
        -0.494,-0.730,-0.931,-1.240,-1.750,-2.351,-3.144,-4.194,-5.359,-6.701,-8.169,
        -9.744,-11.601,-13.608,-15.925,-18.402,-20.911,-23.356,-25.661,-27.658,-29.504,
        -30.966,-32.020,-32.752,-33.149,-33.193,-32.945,-32.451,-31.753,-31.125,-30.544,
        -29.944,-29.532,-29.299,-29.173,-29.302,-29.404,-29.050,-27.895,-25.495,-21.540,
        -16.358,-9.924,-2.500,5.806,15.022,25.102,35.614,45.933,54.963,62.091,67.078,
        69.942,70.926,70.218,68.015,64.560,59.915,54.668,49.232,43.843,39.039,34.656,
        30.492,26.768,23.750,21.114,18.934,16.970,14.696,12.388,10.339,
    ],
    "Beta": [
        0.381,0.294,0.126,-0.010,0.009,0.019,0.131,0.134,0.127,0.073,0.048,0.045,
        0.099,0.199,0.256,0.232,0.095,0.014,-0.080,0.039,0.148,0.336,0.414,0.571,
        0.760,0.979,1.154,1.313,1.465,1.720,2.050,2.353,2.728,3.188,3.648,4.046,
        4.443,4.515,4.350,4.059,3.509,2.788,1.944,0.871,-0.297,-1.495,-2.836,-4.108,
        -5.258,-6.060,-6.696,-7.127,-7.325,-7.228,-6.688,-5.826,-4.807,-3.505,-1.871,
        -0.095,2.035,4.401,6.809,9.430,11.987,14.311,16.594,18.785,20.571,22.015,
        22.863,22.939,22.273,21.096,19.584,17.910,16.121,14.391,12.577,10.696,8.677,
        6.433,3.830,1.306,-1.105,-3.466,-5.851,-8.561,-10.962,-13.129,-15.035,-16.472,
        -18.292,-19.846,-20.531,
    ],
    "Bridge": [
        1.463,0.007,-1.333,-1.633,-0.496,0.449,0.193,-0.431,-1.103,-0.927,0.686,1.277,
        1.444,2.411,1.556,0.602,-0.277,-1.278,-1.676,-0.436,-0.502,0.015,-0.274,
        -0.676,-0.748,0.158,0.418,0.681,1.222,1.055,1.594,2.586,2.534,1.656,-0.189,
        -3.425,-5.329,-7.200,-7.749,-6.824,-6.228,-5.818,-4.419,-3.754,-0.586,3.997,
        8.495,12.525,16.532,19.641,23.009,25.710,27.208,28.165,28.371,26.907,22.953,
        16.986,9.303,-0.301,-10.961,-25.125,-43.670,-63.970,-85.481,-105.288,-120.332,
        -129.268,-131.167,-125.679,-115.645,-100.366,-85.054,-69.823,-50.445,-30.609,
        -11.024,7.269,20.285,28.805,40.047,48.438,56.381,62.015,65.780,68.259,68.956,
        65.280,63.360,60.876,58.291,55.217,46.091,38.920,38.472,
    ],
    "Other": [
        -0.384,-0.308,-0.018,0.156,0.082,-0.003,-0.104,-0.037,0.058,0.123,0.020,-0.053,
        -0.131,-0.273,-0.248,-0.161,-0.007,0.113,0.226,0.021,-0.094,-0.325,-0.406,
        -0.508,-0.668,-0.919,-1.093,-1.231,-1.357,-1.461,-1.657,-1.847,-2.054,-2.312,
        -2.595,-2.729,-3.019,-3.080,-3.039,-3.019,-2.851,-2.608,-2.457,-2.149,-1.982,
        -1.929,-1.816,-1.739,-1.782,-2.018,-2.392,-2.823,-3.220,-3.604,-4.176,-4.701,
        -5.192,-5.745,-6.382,-7.029,-7.949,-8.851,-9.697,-10.749,-11.703,-12.670,
        -13.767,-14.995,-16.171,-17.434,-18.382,-18.988,-18.989,-18.528,-17.924,-17.104,
        -16.138,-15.094,-13.926,-12.721,-11.754,-10.789,-9.949,-9.299,-8.985,-8.949,
        -8.955,-8.605,-8.636,-8.659,-8.789,-9.100,-8.492,-7.828,-7.830,
    ],
}

# Registro de todos os conjuntos de base disponiveis
BASIS_SETS = {
    "DS-dT": {
        "spectra": DS_DT_SPECTRA,
        "combination": DS_DT_COMBINATION,
        "method": "DISICL",
        "description": "Recomendado para proteinas globulares (3 componentes: Alpha, Beta, Coil)",
    },
    "DS5-4": {
        "spectra": DS5_4_SPECTRA,
        "combination": DS5_4_COMBINATION,
        "method": "DISICL",
        "description": "Recomendado para estruturas com loops (6 componentes)",
    },
    "DSSP-1": {
        "spectra": DSSP_1_SPECTRA,
        "combination": None,  # usa classificacao DSSP simplificada
        "method": "DSSP_approx",
        "description": "Aproximacao DSSP via phi/psi (4 componentes: Helix, Beta, Bridge, Other)",
    },
}

BASIS_OPTIONS = list(BASIS_SETS.keys())
DEFAULT_BASIS = "DS-dT"


# =============================================================================
#  DISICL: DEFINICOES DE REGIOES E SEGMENTOS
# =============================================================================
# Extraido de DISICL_prot_det.lib (DISICL v0.97)
# Cada regiao e definida por limites [phi_min, phi_max, psi_min, psi_max]

DISICL_REGIONS = {
    "alfa1":  [(-95, -40, -70, -32)],
    "alfa2":  [(-107, -40, -32, -12)],
    "beta1":  [(-135, -87, 95, 150)],
    "beta2":  [(-175, -135, 95, 136), (-180, -135, 136, 180),
               (-135, -105, 150, 180), (-180, -135, -180.1, -160)],
    "delta":  [(-150, -67, 8, 40), (-150, -107, -32, 8)],
    "delta1": [(-107, -40, -12, 8)],
    "delta2": [(-165, -95, -70, -32)],
    "deltax": [(38, 140, -25, 75)],
    "gamma":  [(55, 95, -100, -40)],
    "gammax": [(-113, -60, 50, 95)],
    "pi":     [(-87, -30, 95, 180), (-100, -60, -180.1, -150)],
    "pix":    [(35, 100, -180.1, -110), (60, 120, 150, 180)],
    "zeta":   [(-170, -113, 50, 95)],
}

# Segmentos: (regiao_residuo_i, regiao_residuo_i+1) -> classe SS
DISICL_SEGMENTS = [
    # 3/10-helix
    ("alfa1", "delta1", "3H"), ("alfa2", "alfa2", "3H"), ("alfa2", "delta", "3H"),
    # Alpha-helix
    ("alfa1", "alfa1", "ALH"), ("alfa1", "alfa2", "ALH"), ("alfa2", "alfa1", "ALH"),
    # Pi-helix
    ("alfa1", "delta2", "PIH"), ("delta2", "delta2", "PIH"),
    ("delta2", "alfa1", "PIH"), ("alfa2", "delta2", "PIH"),
    # Beta-strand
    ("beta2", "beta2", "EBS"), ("beta1", "beta1", "NBS"),
    ("beta1", "beta2", "NBS"), ("beta2", "beta1", "NBS"),
    # Polyproline
    ("pi", "pi", "PP"),
    # Turn type 1
    ("alfa1", "delta", "TI"), ("alfa2", "delta1", "TI"),
    ("delta", "delta", "TI"), ("delta1", "delta", "TI"),
    ("delta1", "delta1", "TI"), ("delta1", "alfa2", "TI"),
    # Turn type 2
    ("pi", "deltax", "TII"),
    # Turn type 8
    ("delta", "zeta", "TVIII"), ("delta1", "zeta", "TVIII"),
    ("alfa2", "zeta", "TVIII"), ("alfa2", "beta1", "TVIII"),
    ("alfa2", "beta2", "TVIII"), ("delta", "gammax", "TVIII"),
    # Gamma-turn
    ("gammax", "alfa2", "GXT"), ("gammax", "delta", "GXT"),
    ("gammax", "delta1", "GXT"), ("gammax", "pi", "GXT"),
    ("pi", "gammax", "GXT"), ("pix", "gamma", "GXT"),
    ("gamma", "pix", "GXT"), ("gamma", "deltax", "GXT"),
    # Schellman-turn
    ("delta", "deltax", "SCH"), ("delta1", "deltax", "SCH"),
    ("deltax", "beta2", "SCH"), ("deltax", "pi", "SCH"),
    # Hairpin
    ("beta1", "deltax", "HP"), ("beta1", "pix", "HP"), ("deltax", "beta1", "HP"),
    # Turn-cap
    ("beta1", "alfa2", "TC"), ("delta", "alfa1", "TC"),
    ("delta", "delta1", "TC"), ("delta", "alfa2", "TC"),
    ("delta1", "alfa1", "TC"), ("delta2", "alfa2", "TC"),
    ("delta2", "delta1", "TC"), ("deltax", "alfa1", "TC"),
    ("deltax", "alfa2", "TC"), ("zeta", "alfa2", "TC"),
    # Helix-cap
    ("alfa2", "delta2", "HC"), ("delta", "delta2", "HC"),
    ("delta1", "delta2", "HC"), ("delta2", "delta", "HC"),
    ("beta1", "alfa1", "HC"), ("beta2", "alfa1", "HC"),
    ("beta2", "alfa2", "HC"), ("pi", "alfa1", "HC"), ("pi", "alfa2", "HC"),
    # Beta-cap
    ("beta1", "pi", "BC"), ("beta2", "pi", "BC"),
    ("pi", "beta1", "BC"), ("pi", "beta2", "BC"),
    # Beta-bulge
    ("pi", "delta", "BU"), ("alfa1", "beta2", "BU"),
    ("delta", "beta2", "BU"), ("beta2", "deltax", "BU"),
    # Left-handed turn
    ("pix", "alfa2", "LTII"), ("pix", "delta", "LTII"), ("pix", "delta1", "LTII"),
    # Left-handed helix
    ("deltax", "deltax", "LHH"),
]


# =============================================================================
#  FUNCOES: PARSING DE PDB
# =============================================================================

def parse_pdb_atoms(pdb_text):
    """
    Extrai atomos do backbone (N, CA, C) de um arquivo PDB.
    Retorna dict: {(chain, resnum): {"N": (x,y,z), "CA": (x,y,z), "C": (x,y,z)}}
    """
    residues = OrderedDict()
    model_done = False

    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            if model_done:
                break
            continue
        if record == "ENDMDL":
            model_done = True
            continue
        if record == "HETATM":
            continue
        if record != "ATOM":
            continue

        # Conformacao alternativa: manter so " " ou "A"
        alt_loc = line[16] if len(line) > 16 else " "
        if alt_loc not in (" ", "A"):
            continue

        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue

        try:
            chain = line[21] if len(line) > 21 else "A"
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        key = (chain, resnum)
        if key not in residues:
            residues[key] = {}
        residues[key][atom_name] = (x, y, z)

    return residues


def clean_pdb_text(pdb_text):
    """Remove HETATM, conformacoes alternativas, mantem 1o modelo."""
    lines = []
    model_done = False
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            if model_done:
                break
            continue
        if record == "ENDMDL":
            model_done = True
            continue
        if record == "HETATM":
            continue
        if record == "ATOM":
            if len(line) > 16 and line[16] not in (" ", "A"):
                continue
        lines.append(line)
    return "\n".join(lines)


# =============================================================================
#  FUNCOES: CALCULO DE ANGULOS DIEDROS
# =============================================================================

def calc_dihedral(p0, p1, p2, p3):
    """
    Calcula o angulo diedro definido por 4 pontos (em graus).
    Usa a convencao IUPAC: phi(C_i-1, N_i, CA_i, C_i).
    """
    b0 = np.array(p1) - np.array(p0)
    b1 = np.array(p2) - np.array(p1)
    b2 = np.array(p3) - np.array(p2)

    # Vetores normais aos planos
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)

    # Normalizar
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm == 0 or n2_norm == 0:
        return None

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Vetor unitario de b1
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0:
        return None
    b1u = b1 / b1_norm

    # m1 = n1 x b1u
    m1 = np.cross(n1, b1u)

    # cos e sin do diedro
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return math.degrees(math.atan2(-y, x))


def compute_dihedrals(residues):
    """
    Calcula phi e psi para cada residuo.
    phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
    psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})
    Retorna: {(chain, resnum): (phi, psi)} apenas para residuos com ambos definidos
    """
    keys = list(residues.keys())
    dihedrals = {}

    for idx, key in enumerate(keys):
        res = residues[key]
        if not all(a in res for a in ("N", "CA", "C")):
            continue

        chain, resnum = key
        phi = None
        psi = None

        # phi: precisa do C do residuo anterior
        if idx > 0:
            prev_key = keys[idx - 1]
            prev_res = residues[prev_key]
            # So calcula se residuo anterior e da mesma cadeia e consecutivo
            if prev_key[0] == chain and "C" in prev_res:
                phi = calc_dihedral(prev_res["C"], res["N"], res["CA"], res["C"])

        # psi: precisa do N do proximo residuo
        if idx < len(keys) - 1:
            next_key = keys[idx + 1]
            next_res = residues[next_key]
            if next_key[0] == chain and "N" in next_res:
                psi = calc_dihedral(res["N"], res["CA"], res["C"], next_res["N"])

        if phi is not None and psi is not None:
            dihedrals[key] = (phi, psi)

    return dihedrals


# =============================================================================
#  FUNCOES: CLASSIFICACAO DISICL
# =============================================================================

def classify_region(phi, psi):
    """Classifica um residuo em uma regiao DISICL baseado em phi/psi."""
    for region_name, bounds_list in DISICL_REGIONS.items():
        for bounds in bounds_list:
            phi_min, phi_max, psi_min, psi_max = bounds
            if phi_min < phi <= phi_max and psi_min < psi <= psi_max:
                return region_name
    return "X"  # nao classificado


def classify_disicl(dihedrals):
    """
    Classifica pares de residuos consecutivos em classes DISICL.
    Retorna dict com contagens de cada classe SS.
    """
    keys = list(dihedrals.keys())
    # Classificar cada residuo em uma regiao
    regions = {}
    for key in keys:
        phi, psi = dihedrals[key]
        regions[key] = classify_region(phi, psi)

    # Classificar pares de residuos consecutivos em segmentos
    ss_counts = {}
    total_segments = 0

    for i in range(len(keys) - 1):
        key1 = keys[i]
        key2 = keys[i + 1]

        # So pares consecutivos da mesma cadeia
        if key1[0] != key2[0]:
            continue
        if key2[1] != key1[1] + 1:
            continue

        reg1 = regions[key1]
        reg2 = regions[key2]

        # Buscar segmento correspondente
        ss_class = "UC"  # unclassified por padrao
        for seg_reg1, seg_reg2, seg_class in DISICL_SEGMENTS:
            if reg1 == seg_reg1 and reg2 == seg_reg2:
                ss_class = seg_class
                break

        ss_counts[ss_class] = ss_counts.get(ss_class, 0) + 1
        total_segments += 1

    # Converter para fracoes (%)
    ss_fractions = {}
    if total_segments > 0:
        for cls, count in ss_counts.items():
            ss_fractions[cls] = (count / total_segments) * 100.0

    return ss_fractions, total_segments


# =============================================================================
#  FUNCOES: CLASSIFICACAO DSSP APROXIMADA (baseada em phi/psi)
# =============================================================================

def classify_dssp_approx(dihedrals):
    """
    Aproximacao da classificacao DSSP usando apenas phi/psi.
    H = helix (phi: -100 a -40, psi: -80 a -5)
    E = strand (phi: -180 a -100, psi: 80 a 180) ou (phi: -180 a -100, psi: -180 a -150)
    T/S/C = rest

    Mapeamento para o basis DSSP-1:
    Helix: 3-Helix + 4-Helix  -> H
    Beta: 5-Helix + Beta-strand -> E
    Bridge: Beta-Bridge -> B (residuos isolados em beta, dificil de detectar sem H-bonds)
    Other: Bend + Turn + Unclassified -> T/S/C
    """
    ss_counts = {"H": 0, "E": 0, "B": 0, "T": 0, "S": 0, "C": 0}
    total = 0

    for key, (phi, psi) in dihedrals.items():
        # Alpha-helix region
        if -100 < phi <= -40 and -80 < psi <= -5:
            ss_counts["H"] += 1
        # Beta-strand region
        elif ((-180 < phi <= -100 and 80 < psi <= 180) or
              (-180 < phi <= -100 and -180 < psi <= -150)):
            ss_counts["E"] += 1
        else:
            ss_counts["C"] += 1
        total += 1

    # Converter para fracoes
    ss_fractions = {}
    if total > 0:
        for cls, count in ss_counts.items():
            if count > 0:
                ss_fractions[cls] = (count / total) * 100.0

    return ss_fractions, total


# =============================================================================
#  FUNCOES: PREDICAO DO ESPECTRO CD
# =============================================================================

def predict_cd_disicl(ss_fractions, basis_name):
    """
    Prediz espectro CD usando classificacao DISICL + conjunto de base.
    ss_fractions: dict com fracoes (%) de cada classe DISICL
    basis_name: "DS-dT" ou "DS5-4"
    """
    bs = BASIS_SETS[basis_name]
    spectra = bs["spectra"]
    combination = bs["combination"]
    basis_names = spectra["basis_names"]
    wavelengths = spectra["wavelengths"]

    # Calcular coeficientes para cada espectro-base
    # Coeficiente_j = sum_k (A_jk * W_k / 100)
    # onde A_jk e a combinacao (0 ou 1) e W_k e a fracao da classe k
    coefficients = {}
    for bname in basis_names:
        coefficients[bname] = 0.0

    for ss_class, fraction in ss_fractions.items():
        if ss_class in combination:
            basis_target = combination[ss_class]
            coefficients[basis_target] += fraction / 100.0

    # Calcular espectro CD: CD(lambda) = sum_j (C_j * B_j(lambda))
    cd_values = []
    for i in range(len(wavelengths)):
        cd = 0.0
        for bname in basis_names:
            cd += coefficients[bname] * spectra[bname][i]
        cd_values.append(cd)

    return wavelengths, cd_values, coefficients


def predict_cd_dssp_approx(ss_fractions, basis_name="DSSP-1"):
    """
    Prediz espectro CD usando classificacao DSSP aproximada.
    """
    spectra = DSSP_1_SPECTRA
    basis_names = spectra["basis_names"]
    wavelengths = spectra["wavelengths"]

    # Mapear classes DSSP para espectros-base
    dssp_to_basis = {
        "H": "Helix",
        "E": "Beta",
        "B": "Bridge",
        "T": "Other",
        "S": "Other",
        "C": "Other",
    }

    coefficients = {bname: 0.0 for bname in basis_names}
    for ss_class, fraction in ss_fractions.items():
        if ss_class in dssp_to_basis:
            basis_target = dssp_to_basis[ss_class]
            coefficients[basis_target] += fraction / 100.0

    cd_values = []
    for i in range(len(wavelengths)):
        cd = 0.0
        for bname in basis_names:
            cd += coefficients[bname] * spectra[bname][i]
        cd_values.append(cd)

    return wavelengths, cd_values, coefficients


# =============================================================================
#  FUNCAO PRINCIPAL DE PREDICAO
# =============================================================================

def predict_spectrum(pdb_text, basis_name="DS-dT", do_clean=True, log=print):
    """
    Pipeline completo: PDB text -> espectro CD predito.
    """
    if do_clean:
        pdb_text = clean_pdb_text(pdb_text)

    # 1. Extrair atomos
    log("Extraindo atomos do backbone...")
    residues = parse_pdb_atoms(pdb_text)
    n_res = len(residues)
    if n_res < 3:
        log(f"Erro: apenas {n_res} residuos encontrados. Minimo: 3.")
        return None
    log(f"  {n_res} residuos com atomos do backbone encontrados")

    # 2. Calcular diedros
    log("Calculando angulos diedros phi/psi...")
    dihedrals = compute_dihedrals(residues)
    n_dih = len(dihedrals)
    if n_dih < 2:
        log(f"Erro: apenas {n_dih} residuos com diedros completos.")
        return None
    log(f"  {n_dih} pares phi/psi calculados")

    bs = BASIS_SETS[basis_name]

    # 3. Classificar estrutura secundaria
    if bs["method"] == "DISICL":
        log(f"Classificando SS via DISICL (metodo detalhado)...")
        ss_fractions, total = classify_disicl(dihedrals)
    else:
        log(f"Classificando SS via aproximacao DSSP (phi/psi)...")
        ss_fractions, total = classify_dssp_approx(dihedrals)

    log(f"  {total} segmentos classificados")

    # Mostrar composicao SS
    if bs["method"] == "DISICL":
        combination = bs["combination"]
        grouped = {}
        for cls, frac in ss_fractions.items():
            if cls in combination:
                group = combination[cls]
            else:
                group = "Other"
            grouped[group] = grouped.get(group, 0) + frac
        for group, frac in sorted(grouped.items(), key=lambda x: -x[1]):
            log(f"  {group}: {frac:.1f}%")
    else:
        for cls, frac in sorted(ss_fractions.items(), key=lambda x: -x[1]):
            log(f"  {cls}: {frac:.1f}%")

    # 4. Predizer espectro CD
    log(f"Calculando espectro CD (base: {basis_name})...")
    if bs["method"] == "DISICL":
        wl, cd, coeffs = predict_cd_disicl(ss_fractions, basis_name)
    else:
        wl, cd, coeffs = predict_cd_dssp_approx(ss_fractions, basis_name)

    log(f"  Espectro gerado: {len(wl)} pontos ({wl[-1]:.0f}-{wl[0]:.0f} nm)")

    # Coeficientes
    for name, coeff in coeffs.items():
        if coeff > 0.001:
            log(f"  Coeficiente {name}: {coeff:.3f}")

    return {
        "wavelengths": wl,
        "cd_values": cd,
        "coefficients": coeffs,
        "ss_fractions": ss_fractions,
        "n_residues": n_res,
        "n_dihedrals": n_dih,
        "basis": basis_name,
    }


# =============================================================================
#  FUNCAO PARA DOWNLOAD DO PDB
# =============================================================================

def fetch_pdb(pdb_id, log=print):
    """Baixa um arquivo PDB do RCSB."""
    pdb_id = pdb_id.upper().strip()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    log(f"Baixando {pdb_id} do RCSB PDB...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode("utf-8", errors="replace")
        if "ATOM" not in data:
            log(f"Arquivo baixado nao contem registros ATOM. Verificar codigo PDB.")
            return None
        log(f"  Download concluido ({len(data)} bytes)")
        return data
    except Exception as e:
        log(f"Nao foi possivel baixar {pdb_id}: {e}")
        return None


def save_combined_csv(results):
    """Gera CSV combinado dos espectros."""
    all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
    names = list(results.keys())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Wavelength_nm"] + names)
    for wl in all_wl:
        row = [wl]
        for name in names:
            r = results[name]
            try:
                idx = r["wavelengths"].index(wl)
                row.append(f"{r['cd_values'][idx]:.4f}")
            except ValueError:
                row.append("")
        writer.writerow(row)

    return output.getvalue()


# =============================================================================
#  CONFIGURACAO DA PAGINA STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="SESCA | CD Spectrum Predictor",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: #fafafa;
        padding: 1.8rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .main-header h1 {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #374151;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .metric-card .unit {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }
    .status-ok { color: #059669; font-size: 0.85rem; font-weight: 500; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## Configuracoes")
    st.markdown("---")

    st.markdown('<span class="status-ok">SESCA Pure-Python pronto</span>', unsafe_allow_html=True)
    st.caption("Nenhuma dependencia externa necessaria")

    st.markdown("---")

    basis = st.selectbox(
        "Conjunto de base espectral",
        options=BASIS_OPTIONS,
        index=BASIS_OPTIONS.index(DEFAULT_BASIS),
        help="\n".join(f"**{k}**: {v['description']}" for k, v in BASIS_SETS.items()),
    )

    clean = st.toggle(
        "Limpar PDB antes de processar",
        value=True,
        help="Remove HETATM (agua, ligantes), conformacoes alternativas, "
             "e mantem apenas o primeiro modelo NMR.",
    )

    st.markdown("---")

    with st.expander("Sobre o algoritmo"):
        st.markdown("""
        **SESCA** (Secondary Structure Estimation for CD Analysis) prediz
        espectros de dicroismo circular a partir de estruturas proteicas.

        **Pipeline:**
        1. Parse PDB → atomos do backbone (N, CA, C)
        2. Calculo de angulos diedros phi/psi
        3. Classificacao DISICL (19 classes detalhadas)
        4. Mapeamento para componentes do espectro-base
        5. Combinacao linear → espectro CD predito

        **Unidades:** Delta-epsilon (10³ deg·cm²/dmol)
        """)

    st.markdown("---")
    st.markdown(
        "<small style='color: #9ca3af;'>"
        "<b>Referencia:</b><br>"
        "Nagy et al., JCTC 15, 5087-5102 (2019)<br>"
        "<a href='https://doi.org/10.1021/acs.jctc.9b00203' style='color: #6b7280;'>"
        "doi: 10.1021/acs.jctc.9b00203</a><br><br>"
        "Nagy & Oostenbrink, JCIM 54, 266-277 (2014)<br>"
        "<a href='https://doi.org/10.1021/ci400541d' style='color: #6b7280;'>"
        "doi: 10.1021/ci400541d</a>"
        "</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>SESCA &mdash; CD Spectrum Predictor</h1>
    <p>Predicao de espectros de Dicroismo Circular a partir de estruturas proteicas (PDB)</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  INPUT
# =============================================================================

tab_upload, tab_rcsb = st.tabs(["Upload de PDB", "Buscar no RCSB PDB"])

uploaded_files = []
pdb_ids = []

with tab_upload:
    files = st.file_uploader(
        "Arraste seus arquivos PDB aqui",
        type=["pdb"],
        accept_multiple_files=True,
    )
    if files:
        uploaded_files = files

with tab_rcsb:
    col1, col2 = st.columns([3, 1])
    with col1:
        pdb_input = st.text_input(
            "Codigos PDB (separados por espaco ou virgula)",
            placeholder="ex: 1UBQ 2GB1 1L2Y",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Buscar", use_container_width=True, type="secondary")

    if pdb_input:
        pdb_ids = [x.strip().upper() for x in pdb_input.replace(",", " ").split() if x.strip()]
        if pdb_ids:
            st.info(f"Estruturas selecionadas: **{', '.join(pdb_ids)}**")


# =============================================================================
#  EXECUCAO
# =============================================================================

has_input = bool(uploaded_files) or bool(pdb_ids)

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_btn = st.button(
        "Executar Predicao",
        use_container_width=True,
        type="primary",
        disabled=not has_input,
    )

if run_btn and has_input:
    st.markdown("---")

    results = {}
    logs = []

    def log_msg(msg):
        logs.append(msg)

    progress = st.progress(0, text="Preparando...")
    all_pdbs = []

    # Coletar arquivos uploaded
    for uf in uploaded_files:
        pdb_text = uf.read().decode("utf-8", errors="replace")
        all_pdbs.append((uf.name.replace(".pdb", ""), pdb_text))

    # Baixar PDBs do RCSB
    for i, pid in enumerate(pdb_ids):
        progress.progress(
            int(10 + 20 * i / max(len(pdb_ids), 1)),
            text=f"Baixando {pid}...",
        )
        pdb_text = fetch_pdb(pid, log=log_msg)
        if pdb_text:
            all_pdbs.append((pid, pdb_text))

    if not all_pdbs:
        st.error("Nenhum arquivo PDB valido para processar.")
        st.stop()

    total = len(all_pdbs)
    for i, (name, pdb_text) in enumerate(all_pdbs):
        pct = int(30 + 60 * i / total)
        progress.progress(pct, text=f"Processando {name}...")
        log_msg(f"\n--- {name} ---")

        result = predict_spectrum(pdb_text, basis_name=basis, do_clean=clean, log=log_msg)
        if result:
            results[name] = result

    progress.progress(95, text="Finalizando...")

    if results:
        csv_data = save_combined_csv(results)

    progress.progress(100, text="Concluido!")

    with st.expander("Log de execucao", expanded=False):
        for line in logs:
            st.text(line)

    if not results:
        st.error("Nenhum espectro foi gerado. Verifique os logs acima.")
        st.stop()

    st.session_state["results"] = results
    st.session_state["csv_data"] = csv_data


# =============================================================================
#  RESULTADOS
# =============================================================================

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    csv_data = st.session_state.get("csv_data", "")

    st.markdown("---")
    st.markdown("## Resultados")

    # Cards com metricas
    n_cols = min(len(results), 4)
    cols = st.columns(n_cols)
    for i, (name, r) in enumerate(results.items()):
        cd = r["cd_values"]
        wl = r["wavelengths"]
        i_min = cd.index(min(cd))
        i_max = cd.index(max(cd))

        with cols[i % n_cols]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{name}</div>'
                f'<div class="value">{min(cd):.2f}</div>'
                f'<div class="unit">Min. CD @ {wl[i_min]:.0f} nm</div>'
                f'<br>'
                f'<div class="value">{max(cd):.2f}</div>'
                f'<div class="unit">Max. CD @ {wl[i_max]:.0f} nm</div>'
                f'<br>'
                f'<div class="unit">{r["n_residues"]} residuos</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grafico
    fig = go.Figure()
    line_colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d"]

    for i, (name, r) in enumerate(results.items()):
        color = line_colors[i % len(line_colors)]
        fig.add_trace(go.Scatter(
            x=r["wavelengths"],
            y=r["cd_values"],
            name=name,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="<b>%{fullData.name}</b><br>"
                          "lambda = %{x:.1f} nm<br>"
                          "CD = %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0, 0, 0, 0.15)")

    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        title=dict(
            text="Espectros de Dicroismo Circular Preditos (SESCA)",
            font=dict(size=16, color="#111827"),
        ),
        xaxis=dict(
            title="Comprimento de onda (nm)",
            gridcolor="#f3f4f6", linecolor="#d1d5db", dtick=10,
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        yaxis=dict(
            title=u"CD (\u0394\u03b5, 10\u00b3 deg\u00b7cm\u00b2/dmol)",
            gridcolor="#f3f4f6", linecolor="#d1d5db",
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(color="#374151", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#d1d5db",
            font=dict(color="#111827", size=12),
        ),
        height=480,
        margin=dict(l=60, r=30, t=50, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Composicao de SS
    with st.expander("Composicao de Estrutura Secundaria", expanded=False):
        for name, r in results.items():
            st.markdown(f"**{name}** ({r['n_residues']} residuos, {r['n_dihedrals']} diedros)")
            coeffs = r["coefficients"]
            coeff_df = pd.DataFrame([
                {"Componente": k, "Coeficiente": f"{v:.4f}", "Fracao (%)": f"{v*100:.1f}"}
                for k, v in sorted(coeffs.items(), key=lambda x: -x[1])
                if v > 0.001
            ])
            st.dataframe(coeff_df, use_container_width=True, hide_index=True)

    # Tabela de dados espectrais
    with st.expander("Tabela de dados espectrais", expanded=False):
        all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
        df_data = {"Wavelength (nm)": all_wl}
        for name, r in results.items():
            wl_map = dict(zip(r["wavelengths"], r["cd_values"]))
            df_data[name] = [wl_map.get(wl, None) for wl in all_wl]
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("### Downloads")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            "CSV Combinado",
            data=csv_data,
            file_name="espectros_CD_combinados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn")
        st.download_button(
            "Grafico Interativo (HTML)",
            data=html_buf.getvalue(),
            file_name="espectro_CD_interativo.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_col3:
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            st.download_button(
                "Grafico (PNG)",
                data=png_bytes,
                file_name="espectro_CD.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption("PNG requer kaleido: pip install kaleido")

else:
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; padding: 3rem; color: #9ca3af;'>"
        "<p style='font-size: 1rem;'>Envie um arquivo PDB ou busque pelo codigo RCSB para comecar</p>"
        "</div>",
        unsafe_allow_html=True,
    )
