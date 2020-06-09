import numpy as np
import pandas as pd
from geopy.distance import distance
from numpy import exp
from scipy.optimize import curve_fit
from lmfit import Model
import statsmodels.api as sm
import pickle

import constants as const

# EXAMPLE ON TIMING CODES
#starttime = time.clock()
#CODES TO BE TIMED HERE
#duration = time.clock() - starttime
#print('shortestPaths takes %.2gs' % duration)

# ======================================================================================================================
def convertSecsToHHMMSS(nSecs):
    '''
    :param nSecs:
    :return:
    '''
    hh = int(nSecs/3600)
    mm = int((nSecs%3600)/60)
    ss = int(nSecs - hh*3600 - mm*60)

    hhStr = ('%d' % hh) if hh >= 10 else ('0%d' % hh)
    mmStr = ('%d' % mm) if mm >= 10 else ('0%d' % mm)
    ssStr = ('%d' % ss) if ss >= 10 else ('0%d' % ss)

    return '%s:%s:%s' % (hhStr, mmStr, ssStr)

# ======================================================================================================================
def makeListFromEnum(myEnum):
    myList = []
    for val in myEnum:
        myList.append(val.name)
    return myList

# ======================================================================================================================
def append2DF(existDF,newDataDict):
    dfNewData = pd.DataFrame(newDataDict, index=[0])
    return existDF.append(dfNewData, ignore_index=True, sort=False)

# ======================================================================================================================
def calcRoadDist(pathPointList):
    '''
    :param pathPointList: e.g. [[106.65256500,10.75125313], [106.65222168,10.75123787], [106.65202332,10.75113773], ...]
    Note that points in pathPointList are in lon,lat order.
    :return:
    '''
    totDist = 0
    for i in range(1,len(pathPointList)):
        crnPointFloat = (pathPointList[i][1], pathPointList[i][0])
        prevPointFloat = (pathPointList[i-1][1], pathPointList[i-1][0])
        dist = distance(prevPointFloat, crnPointFloat).meters
        totDist += dist
    return totDist

# ======================================================================================================================
def calcBinWidth(x):
    '''
    calculates bin width for the list of continuous values x following Freedman-Diaconis formula.
    binwidth = 2 * IQR * n^(-1/3)
    :param x:
    :return:
    '''
    n = len(x)
    Q1 = np.percentile(x,25)
    Q3 = np.percentile(x,75)
    binwidth = 2 * (Q3 - Q1) * n**(-1/3)
    return binwidth

# ======================================================================================================================
def calcGeodesicDist(point1, point2):
    '''
    :param point1: [long,lat]
    :param point2:
    :return:
    '''
    point1Long = point1[0]
    point1Lat = point1[1]
    point2Long = point2[0]
    point2Lat = point2[1]
    return distance((point1Lat,point1Long), (point2Lat,point2Long)).meters

# ======================================================================================================================
def asymetricUnimodal(x,A,B,C):
    return A * x * exp(-B * x**2 + C * x)

# ======================================================================================================================
def curveFit(xObs, yObs, initVals):
    '''
    fits xObs and yObs to asymetricUnimodal function.
    Detail documentation and examples in https://lmfit.github.io/lmfit-py/model.html
    Initial values are critical to convergence.
    Examples: (true values A=0.2, B=0.002, C=0.1)
    dfLens = pd.read_csv('testLens.csv')
    bestVals, covar = utils.curveFit(xObs=dfLens['xObs'].to_list(), yObs=dfLens['yObs'].to_list(),
                                    initVals=[0.1,0.001,0.1])
    :param xObs:
    :param yObs:
    :param initVals:
    :return:
    '''
    bestVals, covar = curve_fit(asymetricUnimodal, xObs, yObs, p0=initVals)
    print('bestVals: {}'.format(bestVals))
    return bestVals, covar

# ======================================================================================================================
def fitAsymetricUnimodal(xObs, yObs, initA, initB, initC):
    '''
    fits xObs and yObs to asymetricUnimodal function.
    Detail documentation and examples in https://lmfit.github.io/lmfit-py/model.html
    Initial values are critical to convergence
    Example: (true values A=0.2, B=0.002, C=0.1)
    dfLens = pd.read_csv('testLens.csv')
    result = utils.modelFit(xObs=dfLens['xObs'].to_list(), yObs=dfLens['yObs'].to_list(), 0.1, 0.001, 0.1)
    :param xObs:
    :param yObs:
    :param initA:
    :param initB:
    :param initC:
    :return:
    '''
    myModel = Model(asymetricUnimodal)
    result = myModel.fit(yObs, x=xObs, A=initA, B=initB, C=initC)
    #print(result.fit_report())
    return result

# ======================================================================================================================
def fitLinReg(X,Y):
    Xconst = sm.add_constant(X)
    model = sm.OLS(Y,Xconst)
    results = model.fit()
    #print(results.params)
    #print(results.summary())
    return results

# ======================================================================================================================
def picklePathLens(shortestPaths, shortestPaths_time, shortestPaths_dist):
    with open(const.shortestPathPkl, 'wb') as f:
        pickle.dump(shortestPaths, f)
    with open(const.shortestTimePkl, 'wb') as f:
        pickle.dump(shortestPaths_time, f)
    with open(const.shortestDistPkl, 'wb') as f:
        pickle.dump(shortestPaths_dist, f)


# ======================================================================================================================
def printStuff(G, dfRoutes):
    print('Nodes')
    for node in G.nodes:
        print([node, G.nodes[node]['StationId'], G.nodes[node]['StationDesc'],
               G.nodes[node]['Lat'], G.nodes[node]['Lng']])
        print(G.nodes[node]['routes'])
        #print('\n')

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print('Edges')
    for edge in G.edges():
        print([edge, G.edges[edge]['nLines'], G.edges[edge]['meanRoadDist'],
               G.edges[edge]['meanTravTime'], G.edges[edge]['nServices'] ])
        print(G.edges[edge]['routes'])
        print('\n')

    print(dfRoutes)