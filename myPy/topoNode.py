import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math
from geopy.distance import distance
import pickle
import os
import multiprocessing
import random

import constants as const
import utilities as utils

def calcGeoDist(G, nodei, nodej):
    latNodei = G.nodes[nodei]['Lat']
    lngNodei = G.nodes[nodei]['Lng']
    latNodej = G.nodes[nodej]['Lat']
    lngNodej = G.nodes[nodej]['Lng']
    return distance((latNodei,lngNodei), (latNodej,lngNodej)).meters

# ======================================================================================================================
def calcStopPairsDist(G, pickleFile):
    if os.path.isfile(pickleFile):
        return pickle.load(open(pickleFile, 'rb'))
    else:
        nodeList = list(G.nodes)
        stopPairsDist = {}
        for i in range(len(nodeList)):
            nodei = nodeList[i]
            for j in range(i + 1, len(nodeList)):
                nodej = nodeList[j]
                dist = calcGeoDist(G, nodei, nodej)
                stopPairsDist['%d_%d' % (nodei, nodej)] = dist
                stopPairsDist['%d_%d' % (nodej, nodei)] = dist

        with open(pickleFile, 'wb') as f:
            pickle.dump(stopPairsDist, f)

        return stopPairsDist

# ======================================================================================================================
def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# ======================================================================================================================
def calcBetwCentralMultip(G, nProcesses, dfShortTime, dfShortDist, dfShortHops):
    # partition node list
    stopParts = partition(list(G.nodes), nProcesses)
    processes = []
    for i in range(nProcesses):
        stopPart = stopParts[i]
        p = multiprocessing.Process(target = calcBetwCentralities,
                                    args = (G, stopPart, dfShortTime, dfShortDist, dfShortHops, i))
        processes.append(p)
        p.start()
        print('calcBetwCentralMultip part %d started, %d stops (out of %d stops)' %
              (i, len(stopPart), len(list(G.nodes))))

    for p in processes:
        p.join()

# ======================================================================================================================
def calcBetwCentralities(G, stopsList, dfShortTime, dfShortDist, dfShortHops, part):
    '''
    columns in dfShortTimePath are 'nodei', 'nodej', 'path', 'value', 'pathStr', 'ijDist'
    :param G:
    :param dfShortTime:
    :return:
    '''
    # CALCULATES GEOGRAPHIC DISTANCE BETWEEN NODEI AND NODEJ FOR ALL PAIRS
    stopPairsDist = calcStopPairsDist(G, '%s/stopPairsDist.pkl' % const.picklesFolder)

    # dfShortTime['ijDist'] = dfShortTime.apply(lambda row: calcGeoDist(G, row['nodei'], row['nodej']), axis=1)
    dfShortTime['stopPairs'] = dfShortTime.apply(lambda row: '%s_%s' % (row['nodei'], row['nodej']), axis=1)
    dfShortTime['ijDist'] = dfShortTime['stopPairs'].map(stopPairsDist)

    # dfShortDist['ijDist'] = dfShortDist.apply(lambda row: calcGeoDist(G, row['nodei'], row['nodej']), axis=1)
    dfShortDist['stopPairs'] = dfShortDist.apply(lambda row: '%s_%s' % (row['nodei'], row['nodej']), axis=1)
    dfShortDist['ijDist'] = dfShortDist['stopPairs'].map(stopPairsDist)

    # dfShortHops['ijDist'] = dfShortHops.apply(lambda row: calcGeoDist(G, row['nodei'], row['nodej']), axis=1)
    dfShortHops['stopPairs'] = dfShortHops.apply(lambda row: '%s_%s' % (row['nodei'], row['nodej']), axis=1)
    dfShortHops['ijDist'] = dfShortHops['stopPairs'].map(stopPairsDist)

    dfShortTimeO300m = dfShortTime.loc[dfShortTime['ijDist'] > const.maxWalkDist]
    dfShortDistO300m = dfShortDist.loc[dfShortDist['ijDist'] > const.maxWalkDist]
    dfShortHopsO300m = dfShortHops.loc[dfShortHops['ijDist'] > const.maxWalkDist]

    # calculates betweenness centrality by time, by distance, and by number of hops
    betwCentrals = []
    ndCounts = 0
    for nodev in stopsList:
    #for nodev in G.nodes:
        nodevStr = '_%d_' % nodev
        nPathsByTime = len(dfShortTimeO300m.loc[dfShortTimeO300m['pathStr'].str.contains(nodevStr)].index)
        nPathsByDist = len(dfShortDistO300m.loc[dfShortDistO300m['pathStr'].str.contains(nodevStr)].index)
        nPathsByHops = len(dfShortHopsO300m.loc[dfShortHopsO300m['pathStr'].str.contains(nodevStr)].index)
        betwCentrals.append([nodev, nPathsByTime, nPathsByDist, nPathsByHops])

        ndCounts += 1
        if ndCounts % 10 == 0:
            print('part %d, %d stations completed (out of %d stops)' % (part, ndCounts, len(stopsList)))

    dfBetwCentrals = pd.DataFrame(betwCentrals, columns = ['nodev', 'nPathsTime', 'nPathsDist', 'nPathsHops'])

    '''
    cbHopsMax = dfBetwCentrals['nPathsHops'].max()
    cbHopsMin = dfBetwCentrals['nPathsHops'].min()
    cbDistMax = dfBetwCentrals['nPathsDist'].max()
    cbDistMin = dfBetwCentrals['nPathsDist'].min()
    cbTimeMax = dfBetwCentrals['nPathsTime'].max()
    cbTimeMin = dfBetwCentrals['nPathsTime'].min()

    dfBetwCentrals[const.GNodeAttribs.cbHops.name] = \
        dfBetwCentrals.apply(lambda row: (row['nPathsHops'] - cbHopsMin) / (cbHopsMax - cbHopsMin), axis=1)
    dfBetwCentrals[const.GNodeAttribs.cbDist.name] = \
        dfBetwCentrals.apply(lambda row: (row['nPathsDist'] - cbDistMin) / (cbDistMax - cbDistMin), axis=1)
    dfBetwCentrals[const.GNodeAttribs.cbTime.name] = \
        dfBetwCentrals.apply(lambda row: (row['nPathsTime'] - cbTimeMin) / (cbTimeMax - cbTimeMin), axis=1)
    '''
    nAllStops = len(G.nodes)
    normalisedFactor = (nAllStops-1)*(nAllStops-2)
    dfBetwCentrals[const.GNodeAttribs.cbHops.name] = \
        dfBetwCentrals.apply(lambda row: row['nPathsHops'] / normalisedFactor, axis=1)
    dfBetwCentrals[const.GNodeAttribs.cbDist.name] = \
        dfBetwCentrals.apply(lambda row: row['nPathsDist'] / normalisedFactor, axis=1)
    dfBetwCentrals[const.GNodeAttribs.cbTime.name] = \
        dfBetwCentrals.apply(lambda row: row['nPathsTime'] / normalisedFactor, axis=1)

    dfBetwCentrals.to_csv('%s/nodeCentrals/bc_%d.csv' % (const.outputsFolder, part), index=False)
    print('part %d written' % part)

    #return dfBetwCentrals

# ======================================================================================================================
def calcProxDensity(G, dfShortTime):
    '''
    columns in dfShortTimePath are 'nodei', 'nodej', 'path', 'value', 'pathStr', 'ijDist'
    :param G:
    :param dfShortTime:
    :return:
    '''
    nSecs30mins = 30 * 60
    nSecs45mins = 45 * 60
    nSecs60mins = 60 * 60
    nSecs90mins = 90 * 60

    # CALCULATES GEOGRAPHIC DISTANCE BETWEEN NODEI AND NODEJ FOR ALL PAIRS
    dfShortTime['ijDist'] = dfShortTime.apply(lambda row: calcGeoDist(G, row['nodei'], row['nodej']), axis=1)
    dfO300m = dfShortTime.loc[dfShortTime['ijDist']>const.maxWalkDist]

    toFrom = []
    ndCounts = 0
    for nodev in G.nodes:
        # under 30 minutes
        nNdsFrV_U30 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs30mins)].count()
        nNdsToV_U30 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs30mins)].count()
        ndsFrV_U30 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs30mins)]
        ndsToV_U30 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs30mins)]

        # 30 mins to 60 mins
        nNdsFrV_3060 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                            (dfO300m['value'] > nSecs30mins) &
                                            (dfO300m['value'] <= nSecs60mins)].count()
        nNdsToV_3060 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                            (dfO300m['value'] > nSecs30mins) &
                                            (dfO300m['value'] <= nSecs60mins)].count()
        ndsFrV_3060 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                           (dfO300m['value'] > nSecs30mins) & (dfO300m['value'] <= nSecs60mins)]
        ndsToV_3060 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                           (dfO300m['value'] > nSecs30mins) & (dfO300m['value'] <= nSecs60mins)]

        # 60 mins to 90 mins
        nNdsFrV_6090 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                            (dfO300m['value'] > nSecs60mins) &
                                            (dfO300m['value'] <= nSecs90mins)].count()
        nNdsToV_6090 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                            (dfO300m['value'] > nSecs60mins) &
                                            (dfO300m['value'] <= nSecs90mins)].count()
        ndsFrV_6090 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                           (dfO300m['value'] > nSecs60mins) & (dfO300m['value'] <= nSecs90mins)]
        ndsToV_6090 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                           (dfO300m['value'] > nSecs60mins) & (dfO300m['value'] <= nSecs90mins)]

        # under 60 minutes
        nNdsFrV_U60 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs60mins)].count()
        nNdsToV_U60 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs60mins)].count()
        ndsFrV_U60 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs60mins)]
        ndsToV_U60 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs60mins)]

        # under 90 minutes
        nNdsFrV_U90 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs90mins)].count()
        nNdsToV_U90 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                           (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs90mins)].count()
        ndsFrV_U90 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs90mins)]
        ndsToV_U90 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) &
                                          (dfO300m['value'] > 0) & (dfO300m['value'] <= nSecs90mins)]

        # over 90 minutes
        nNdsFrV_O90 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) & (dfO300m['value'] > nSecs90mins)].count()
        nNdsToV_O90 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) & (dfO300m['value'] > nSecs90mins)].count()
        ndsFrV_O90 = dfO300m['nodej'].loc[(dfO300m['nodei'] == nodev) & (dfO300m['value'] > nSecs90mins)]
        ndsToV_O90 = dfO300m['nodei'].loc[(dfO300m['nodej'] == nodev) & (dfO300m['value'] > nSecs90mins)]

        toFrom.append([nodev,
                       nNdsFrV_U30, nNdsFrV_U60, nNdsFrV_U90, nNdsFrV_O90, nNdsFrV_3060, nNdsFrV_6090,
                       nNdsToV_U30, nNdsToV_U60, nNdsToV_U90, nNdsToV_O90, nNdsToV_3060, nNdsToV_6090,
                       ndsFrV_U30, ndsFrV_U60, ndsFrV_U90, ndsFrV_O90, ndsFrV_3060, ndsFrV_6090,
                       ndsToV_U30, ndsToV_U60, ndsToV_U90, ndsToV_O90, ndsToV_3060, ndsToV_6090])

        ndCounts += 1
        if ndCounts % 100 == 0:
            print('%d stations completed' % ndCounts)

    dfTempAccess = pd.DataFrame(toFrom,
                                columns=['nodev',
                                         'frV_u30', 'frV_u60', 'frV_u90', 'frV_o90', 'frV_3060', 'frV_6090',
                                         'toV_u30', 'toV_u60', 'toV_u90', 'toV_o90', 'toV_3060', 'toV_6090',
                                         'ndfrV_u30', 'ndfrV_u60', 'ndfrV_u90', 'ndfrV_o90', 'ndfrV_3060', 'ndfrV_6090',
                                         'ndtoV_u30', 'ndtoV_u60', 'ndtoV_u90', 'ndtoV_o90', 'ndtoV_3060', 'ndtoV_6090'])

    dfTempAccess['frV_u30'] = dfTempAccess['frV_u30'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['frV_u60'] = dfTempAccess['frV_u60'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['frV_u90'] = dfTempAccess['frV_u90'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['frV_o90'] = dfTempAccess['frV_o90'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['frV_3060'] = dfTempAccess['frV_3060'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['frV_6090'] = dfTempAccess['frV_6090'].apply(lambda x: x / (len(G.nodes) - 1))

    dfTempAccess['toV_u30'] = dfTempAccess['toV_u30'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['toV_u60'] = dfTempAccess['toV_u60'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['toV_u90'] = dfTempAccess['toV_u90'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['toV_o90'] = dfTempAccess['toV_o90'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['toV_3060'] = dfTempAccess['toV_3060'].apply(lambda x: x / (len(G.nodes) - 1))
    dfTempAccess['toV_6090'] = dfTempAccess['toV_6090'].apply(lambda x: x / (len(G.nodes) - 1))

    dfu30 = dfTempAccess[['nodev', 'frV_u30', 'toV_u30', 'ndfrV_u30', 'ndtoV_u30']]
    dfu60 = dfTempAccess[['nodev', 'frV_u60', 'toV_u60', 'ndfrV_u60', 'ndtoV_u60']]
    dfu90 = dfTempAccess[['nodev', 'frV_u90', 'toV_u90', 'ndfrV_u90', 'ndtoV_u90']]
    dfo90 = dfTempAccess[['nodev', 'frV_o90', 'toV_o90', 'ndfrV_o90', 'ndtoV_o90']]
    df3060 = dfTempAccess[['nodev', 'frV_3060', 'toV_3060', 'ndfrV_3060', 'ndtoV_3060']]
    df6090 = dfTempAccess[['nodev', 'frV_6090', 'toV_6090', 'ndfrV_6090', 'ndtoV_6090']]

    return dfu30, dfu60, dfu90, dfo90, df3060, df6090

# ======================================================================================================================
def calcBetweennessCentrality(G):
    cbHops = nx.betweenness_centrality(G, normalized=True)
    print('finished cbHops')
    cbDist = nx.betweenness_centrality(G, weight='meanRoadDist', normalized=True)
    print('finished cbDist')
    cbTime = nx.betweenness_centrality(G, weight='meanRoadTime', normalized=True)
    print('finished cbTime')

    dfcb = pd.DataFrame.from_dict(cbHops, orient='index', columns=['cbHopsNorm'])
    dfcb['cbDistNorm'] = dfcb.index.map(cbDist)
    dfcb['cbTimeNorm'] = dfcb.index.map(cbTime)

    return dfcb

# ======================================================================================================================
def calcTargetClosenessCentrality_v2(G):
    ccHopsTarget = nx.closeness_centrality(G)
    ccDistTarget = nx.closeness_centrality(G, distance='meanRoadDist')
    ccTimeTarget = nx.closeness_centrality(G, distance='meanTravTime')

    dfccTarget = pd.DataFrame.from_dict(ccHopsTarget, orient='index', columns=['ccHopsNorm'])
    dfccTarget['ccDistNorm'] = dfccTarget.index.map(ccDistTarget)
    dfccTarget['ccTimeNorm'] = dfccTarget.index.map(ccTimeTarget)

    return dfccTarget

# ======================================================================================================================
def calcSourceClosenessCentrality(G, shortHops, shortDist, shortTime):
    '''
    calculates normalised closeness centralities by source from results of shortest distances
    - ignoring identical nodes
    - ignoring nodes within walk distance to each other
    Run of BUDF
    :param G:
    :param shortHops:
    :param shortDist:
    :param shortTime:
    :return:
    '''

    lenKey = 0
    pathKey = 1

    distFrom = []
    for iNode in G.nodes():
        lenHops = 0
        lenDist = 0
        lenTime = 0
        for jNode in G.nodes():
            if iNode==jNode: continue

            nodeiCoord = [G.nodes[iNode][const.GNodeAttribs.Lng.name], G.nodes[iNode][const.GNodeAttribs.Lat.name]]
            nodejCoord = [G.nodes[jNode][const.GNodeAttribs.Lng.name], G.nodes[jNode][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(nodeiCoord, nodejCoord)  # distance in metres
            if dist <= const.maxWalkDist: continue

            lenHops += shortHops[iNode][lenKey][jNode]
            lenDist += shortDist[iNode][lenKey][jNode]
            lenTime += shortTime[iNode][lenKey][jNode]

        distFrom.append([iNode, lenHops, lenDist, lenTime])

    dfDistFrom = pd.DataFrame(distFrom, columns=['node', 'lenHops', 'lenDist', 'lenTime'])
    dfDistFrom['lenHopsNorm'] = dfDistFrom['lenHops'].apply(lambda x: len(G.nodes) / x)
    dfDistFrom['lenDistNorm'] = dfDistFrom['lenDist'].apply(lambda x: len(G.nodes) / x)
    dfDistFrom['lenTimeNorm'] = dfDistFrom['lenTime'].apply(lambda x: len(G.nodes) / x)

    return dfDistFrom

# ======================================================================================================================
def calcTargetClosenessCentrality(G, shortHops, shortDist, shortTime):
    '''
    calculates normalised closeness centralities by target from results of shortest distances
    - ignoring identical nodes
    - ignoring nodes within walk distance to each other
    Run of BUDF
    :param G:
    :param shortHops:
    :param shortDist:
    :param shortTime:
    :return:
    '''

    lenKey = 0
    pathKey = 1

    distTo = []
    for jNode in G.nodes():
        lenHops = 0
        lenDist = 0
        lenTime = 0
        for iNode in G.nodes():
            if iNode==jNode: continue

            nodeiCoord = [G.nodes[iNode][const.GNodeAttribs.Lng.name], G.nodes[iNode][const.GNodeAttribs.Lat.name]]
            nodejCoord = [G.nodes[jNode][const.GNodeAttribs.Lng.name], G.nodes[jNode][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(nodeiCoord, nodejCoord)  # distance in metres
            if dist <= const.maxWalkDist: continue

            lenHops += shortHops[iNode][lenKey][jNode]
            lenDist += shortDist[iNode][lenKey][jNode]
            lenTime += shortTime[iNode][lenKey][jNode]

        distTo.append([jNode, lenHops, lenDist, lenTime])

    dfDistTo = pd.DataFrame(distTo, columns=['node', 'lenHops', 'lenDist', 'lenTime'])
    dfDistTo['lenHopsNorm'] = dfDistTo['lenHops'].apply(lambda x: len(G.nodes) / x)
    dfDistTo['lenDistNorm'] = dfDistTo['lenDist'].apply(lambda x: len(G.nodes) / x)
    dfDistTo['lenTimeNorm'] = dfDistTo['lenTime'].apply(lambda x: len(G.nodes) / x)

    return dfDistTo

# ======================================================================================================================
def plotObsvsPredict(Xobs, Yobs, Xpred, Ypred, xLabel, yLabel, filename, ax):
    #fig = plt.figure(figsize=(15, 10))

    ax.plot(Xobs, Yobs, '.', color='black')
    ax.plot(Xpred, Ypred, '-', color='blue')
    ax.set_xlabel(xLabel, fontsize=11)
    ax.set_ylabel(yLabel, fontsize=11)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xticks([2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels([2,3,4,5,6,7,8,9,10])

    ax.grid(b=True, which='both', axis='both', color='black', linestyle=':', linewidth=.5)

    #plt.tight_layout()
    #fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    #plt.clf()

    '''
    fig, ax = plt.subplots()
    ax.plot(Xobs, Yobs, '.', color='black')
    ax.plot(Xpred, Ypred, '-', color='blue')
    ax.set_yscale('log')
    plt.xlabel(xLabel, fontsize=11)
    plt.ylabel(yLabel, fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.clf()
    '''

# ======================================================================================================================
def fitDegDistribPower(nodeDegree, description, axes):
    '''
    fits node degree following a power law
    :param nodeDegree: can be either node degree or node strength (in terms of nLines or nServices)
    :return:
    '''
    dfDegree = pd.DataFrame(nodeDegree, columns=['node', 'degree'])
    uValCounts = dfDegree['degree'].value_counts()
    pK = uValCounts.values/sum(uValCounts.values)
    k = np.asarray(uValCounts.index.to_list())

    log_pK = np.log(pK)
    log_k = np.log(k)
    results = utils.fitLinReg(log_k,log_pK)

    olsConst = results.params[0]
    olsCoeff = results.params[1]

    summaryFilename = '%s/%s_olsSummary.txt' % (const.nodeDegDistribFolder, description)
    loglogPlotFilename = '%s/%s_loglogPlot.png' % (const.nodeDegDistribFolder, description)
    plotFilename = '%s/%s_plot.png' % (const.nodeDegDistribFolder, description)
    with open(summaryFilename,'w') as fh:
        fh.write(results.summary().as_text())

    gamma = -olsCoeff
    C = math.exp(olsConst)

    kfit = np.linspace(min(k), max(k), num=10)
    pKfit = C * np.power(kfit, -gamma)
    plotObsvsPredict(k, pK, kfit, pKfit, 'Node degree, k', 'P(k)', plotFilename, ax=axes)

    #log_k_fit = np.linspace(min(log_k), max(log_k), num=10)
    #log_pK_fit = olsCoeff * log_k_fit + olsConst
    #plotObsvsPredict(log_k, log_pK, log_k_fit, log_pK_fit, 'log(k)', 'log(P(k))', loglogPlotFilename)
    print('the power law used to fit %s distribution is p(k) = %.3f*k^(-%.3f)' % (description, C, gamma))



# ======================================================================================================================
def fitDegDistribExponential(nodeDegree, description):
    '''
    :param nodeDegree:
    :param description:
    :return:
    '''
    dfDegree = pd.DataFrame(nodeDegree, columns=['node', 'degree'])
    uValCounts = dfDegree['degree'].value_counts()
    pK = uValCounts.values / sum(uValCounts.values)
    k = np.asarray(uValCounts.index.to_list())

    log_pK = np.log(pK)
    results = utils.fitLinReg(k, log_pK)

    olsConst = results.params[0]
    olsCoeff = results.params[1]

    summaryFilename = '%s/%s_olsSummary.txt' % (const.nodeDegDistribFolder, description)
    logPlotFilename = '%s/%s_logPlot.png' % (const.nodeDegDistribFolder, description)
    plotFilename = '%s/%s_plot.png' % (const.nodeDegDistribFolder, description)
    with open(summaryFilename, 'w') as fh:
        fh.write(results.summary().as_text())

    kHat = -1./olsCoeff
    C = math.exp(olsConst)

    kfit = np.linspace(min(k), max(k), num=10)
    pKfit = C * np.exp(-kfit/kHat)
    plotObsvsPredict(k, pK, kfit, pKfit, 'Node degree, k', 'P(k)', plotFilename)

    log_pK_fit = olsCoeff * kfit + olsConst
    plotObsvsPredict(k, log_pK, kfit, log_pK_fit, 'Node degree, k', 'log(P(k))', logPlotFilename)

    print('the exponential law used to fit %s distribution is p(k) = %.3f*e^(-k/%.3f)' % (description, C, kHat))


# ======================================================================================================================
def fitNdStrengthDegPower(nodeStrength, nodeDegree, xLabel, yLabel, filename):
    # scatter plot of nodeStrength vs nodeDegree
    dfDegree = pd.DataFrame(nodeDegree, columns=['node', 'degree'])
    #dfDegree.to_csv('dfDegree_prior.csv')
    dfStrength = pd.DataFrame(nodeStrength, columns=['node', 'strength'])
    #dfStrength.to_csv('dfStrength.csv')
    dfDegree = dfDegree.join(dfStrength.set_index('node'), on='node')
    #dfDegree.to_csv('dfDegree_poster.csv')
    df = dfDegree.groupby('degree')['strength'].mean()
    #df.to_csv('dfGrouped.csv')
    print(df.index)
    print(df.values)
    plt.plot(df.index,df.values,'.', color='black')
    plt.xlabel(xLabel, fontsize=11)
    plt.ylabel(yLabel, fontsize=11)
    # plt.show()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=1)
    plt.clf()




# WILL TAKE A VERY LONG TIME - DO NOT USE ==============================================================================
def calcBetweennessCentrality(G, shortHops, shortDist, shortTime):
    '''
    calculates betweenness centrality for all pair of nodes that are
    - not identical
    - not within walk distance to each other
    :param shortHops:
    :param shortDist:
    :param shortTime:
    :return:
    '''

    lenKey = 0
    pathKey = 1

    cb = []
    for node in G.nodes:
        ndCbHops = 0
        ndCbDist = 0
        ndCbTime = 0
        for iNode in G.nodes:
            if iNode==node: continue
            for jNode in G.nodes():
                if jNode==node or jNode==iNode: continue

                nodeiCoord = [G.nodes[iNode][const.GNodeAttribs.Lng.name], G.nodes[iNode][const.GNodeAttribs.Lat.name]]
                nodejCoord = [G.nodes[jNode][const.GNodeAttribs.Lng.name], G.nodes[jNode][const.GNodeAttribs.Lat.name]]
                dist = utils.calcGeodesicDist(nodeiCoord, nodejCoord)  # distance in metres
                if dist <= const.maxWalkDist: continue

                sPathHops = shortHops[iNode][pathKey][jNode]
                sPathDist = shortDist[iNode][pathKey][jNode]
                sPathTime = shortTime[iNode][pathKey][jNode]
                if node in sPathHops:
                    ndCbHops += 1
                if node in sPathDist:
                    ndCbDist += 1
                if node in sPathTime:
                    ndCbTime += 1

        cb.append([node, ndCbHops, ndCbDist, ndCbTime])

    dfCb = pd.DataFrame(cb, columns=['node', 'cbHops', 'cbDist', 'cbTime'])
    cbHopsMax = dfCb['CbHops'].max()
    cbHopsMin = dfCb['CbHops'].min()
    cbDistMax = dfCb['CbDist'].max()
    cbDistMin = dfCb['CbDist'].min()
    cbTimeMax = dfCb['CbTime'].max()
    cbTimeMin = dfCb['CbTime'].min()

    dfCb['cbHopsNorm'] = dfCb.apply(lambda row: (row['CbHops'] - cbHopsMin) / (cbHopsMax - cbHopsMin), axis=1)
    dfCb['cbDistNorm'] = dfCb.apply(lambda row: (row['CbDist'] - cbDistMin) / (cbDistMax - cbDistMin), axis=1)
    dfCb['cbTimeNorm'] = dfCb.apply(lambda row: (row['CbTime'] - cbTimeMin) / (cbTimeMax - cbTimeMin), axis=1)

    return dfCb