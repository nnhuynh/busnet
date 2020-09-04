import pickle
import pandas as pd
import time
import multiprocessing
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os
from enum import Enum

import constants as const
import utilities as utils
import geoPlotter
import topoPostproc

# ======================================================================================================================
def getPathStops(pathNodes, ndIDict):
    return [int(ndIDict[node].split('_')[2]) for node in pathNodes]

def countTransfers(path, dfAllTransfers):
    nTransfers = 0
    for i in range(len(path) - 1):
        thisNdFr = path[i]
        thisNdTo = path[i + 1]
        nRowsFound = len(dfAllTransfers.loc[(dfAllTransfers[const.dfAllTransfersCols.nodeFr.name] == thisNdFr) &
                                            (dfAllTransfers[const.dfAllTransfersCols.nodeTo.name] == thisNdTo)].index)
        nTransfers += nRowsFound
    return nTransfers

# ======================================================================================================================
def consolidateOutputFiles(shortestPathsFiles, dfAllTransfers, ndIDict):
    allShortestPaths = pd.DataFrame()
    for file in shortestPathsFiles:
        print('reading file %s... ' % file)
        shortestPaths = pickle.load(open(file, 'rb'))
        dfShortestPaths = pd.DataFrame(shortestPaths,
                                       columns=['stopFr', 'stopTo', 'timeFr', 'timeTo', 'path', 'nTransfers'])
        # excludes paths that have only 1 walk edge, i.e. stopFr and stopTo are within walk distance
        dfTransitPaths = dfShortestPaths.loc[dfShortestPaths['nTransfers'] != -1]
        #dfTransitPaths['nTransfers'] = dfTransitPaths['path'].apply(countTransfers, dfAllTransfers = dfAllTransfers)
        dfTransitPaths['pathStops'] = dfTransitPaths['path'].apply(getPathStops, ndIDict=ndIDict)
        dfTransitPaths['tripTime'] = dfTransitPaths.apply(lambda row: row['timeTo'] - row['timeFr'], axis=1)

        allShortestPaths = pd.concat([allShortestPaths, dfTransitPaths[['stopFr', 'stopTo', 'tripTime', 'pathStops']]])

    return allShortestPaths

# ======================================================================================================================
def isStopInPath(stopList, stop):
    if stop in stopList[1:-1]: return True
    else: return False

# ======================================================================================================================
def calcBetwCentral(stops, normalisedFactor, dfAllShortestPaths, procID, outputFolder):
    '''
    :param stops:
    :param dfAllShortestPaths: has 4 columns 'stopFr', 'stopTo', 'tripTime', 'pathStops' and includes only path between
    stop pairs that are NOT within walk distance
    :return:
    '''
    '''
    # example
    mylist = []
    mylist.append([1, 2, 3, [12, 14, 314, 1554, 15]])
    mylist.append([11, 12, 13, [102, 104, 314, 15540, 105, 550]])
    mylist.append([21, 22, 23, [120, 140, 314, 1554, 15, 55]])
    mylist.append([31, 32, 33, [120, 15]])

    df = pd.DataFrame(mylist, columns = ['col1', 'col2', 'col3', 'stopStr'])

    results = []
    stop = 15
    df['in'] = df['stopStr'].apply(isStopInPath, stop=stop)
    nPairs = len(df.loc[df['in']==True].index) / 5
    print('\nstop %d, %d' % (stop, nPairs))
    print(df[['col1', 'stopStr', 'in']])
    results.append([stop, nPairs])

    stop = 15000
    df['in'] = df['stopStr'].apply(isStopInPath, stop=stop)
    nPairs = len(df.loc[df['in'] == True].index) / 5
    print('\nstop %d, %d' % (stop, nPairs))
    print(df[['col1', 'stopStr', 'in']])
    results.append([stop, nPairs])

    stop = 314
    df['in'] = df['stopStr'].apply(isStopInPath, stop=stop)
    nPairs = len(df.loc[df['in'] == True].index) / 5
    print('\nstop %d, %d' % (stop, nPairs))
    print(df[['col1', 'stopStr', 'in']])
    results.append([stop, nPairs])

    print()
    print(results)
    '''

    betwCentral = []

    stopCounts = 0
    starttime = time.perf_counter()
    for stop in stops:
        dfAllShortestPaths['viaStop'] = dfAllShortestPaths['pathStops'].apply(isStopInPath, stop=stop)
        fracPathsViaStop = len(dfAllShortestPaths.loc[dfAllShortestPaths['viaStop']==True].index) / normalisedFactor
        betwCentral.append([stop, fracPathsViaStop])

        stopCounts += 1
        if stopCounts % 10 == 0:
            print('process %d - %d/%d stops completed, took %.4g mins' % (procID, stopCounts, len(stops),
                                                                          (time.perf_counter() - starttime)/60))

    dfBetwCentral = pd.DataFrame(betwCentral, columns = ['stop', 'normBetwCentral'])
    dfBetwCentral.to_csv('%s/ttgBC_%d.csv' % (outputFolder, procID), index=False)
    #return dfBetwCentral


# ======================================================================================================================
def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def calcBetwCentralMultip(allStops, nProcesses, dfAllShortestPaths, outputFolder):
    # normalised factor for directed graph (https://en.wikipedia.org/wiki/Betweenness_centrality)
    nAllStops = len(allStops)
    normalisedFactor = (nAllStops-1)*(nAllStops-2)

    stopParts = partition(allStops, nProcesses)
    processes = []
    for i in range(nProcesses):
        stops = stopParts[i]
        p = multiprocessing.Process(target = calcBetwCentral,
                                    args = (stops, normalisedFactor, dfAllShortestPaths, i, outputFolder))
        processes.append(p)
        p.start()
        print('calcBetwCentralMultip procID %d started for %d stops' % (i, len(stops)))

    for p in processes:
        p.join()

# ======================================================================================================================
def consolidateBetwCentrals(nProcesses, outputFolder):
    # consolidates betweenness centrality results from multiprocessing
    dfbcAll = pd.DataFrame()
    for i in range(nProcesses):
        dfbcPart = pd.read_csv('%s/ttgBC_%d.csv' % (outputFolder, i))
        dfbcAll = pd.concat([dfbcAll, dfbcPart])
    dfbcAll.to_csv('%s/ttgBC_All.csv' % outputFolder, index=False)

# ======================================================================================================================
def calcProxDensity(allStops, dfAllShortestPaths):
    '''
    :param allStops:
    :param dfAllShortestPaths: has 4 columns 'stopFr', 'stopTo', 'tripTime', 'pathStops' and includes only path between
    stop pairs that are NOT within walk distance
    :return:
    '''

    nSecs30mins = 30 * 60
    nSecs60mins = 60 * 60
    nSecs90mins = 90 * 60

    proxDens_U30 = []
    proxDens_3060 = []
    proxDens_6090 = []
    proxDens_U60 = []
    proxDens_U90 = []
    proxDens_O90 = []

    denominator = len(allStops) - 1
    for stopv in allStops:
        # under 30 minutes
        nStopsFrV_U30 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs30mins)].index) / denominator
        nStopsToV_U30 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs30mins)].index) / denominator
        stopsFrV_U30 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs30mins)].values.tolist()
        stopsToV_U30 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs30mins)].values.tolist()

        # 30 mins to 60 mins
        nStopsFrV_3060 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                    (dfAllShortestPaths['tripTime'] > nSecs30mins) &
                                                    (dfAllShortestPaths['tripTime'] <= nSecs60mins)].index) / denominator
        nStopsToV_3060 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                    (dfAllShortestPaths['tripTime'] > nSecs30mins) &
                                                    (dfAllShortestPaths['tripTime'] <= nSecs60mins)].index) / denominator
        stopsFrV_3060 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                         (dfAllShortestPaths['tripTime'] > nSecs30mins) &
                                                         (dfAllShortestPaths['tripTime'] <= nSecs60mins)].values.tolist()
        stopsToV_3060 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                         (dfAllShortestPaths['tripTime'] > nSecs30mins) &
                                                         (dfAllShortestPaths['tripTime'] <= nSecs60mins)].values.tolist()

        # 60 mins to 90 mins
        nStopsFrV_6090 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                    (dfAllShortestPaths['tripTime'] > nSecs60mins) &
                                                    (dfAllShortestPaths['tripTime'] <= nSecs90mins)].index) / denominator
        nStopsToV_6090 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                    (dfAllShortestPaths['tripTime'] > nSecs60mins) &
                                                    (dfAllShortestPaths['tripTime'] <= nSecs90mins)].index) / denominator
        stopsFrV_6090 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                         (dfAllShortestPaths['tripTime'] > nSecs60mins) &
                                                         (dfAllShortestPaths['tripTime'] <= nSecs90mins)].values.tolist()
        stopsToV_6090 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                         (dfAllShortestPaths['tripTime'] > nSecs60mins) &
                                                         (dfAllShortestPaths['tripTime'] <= nSecs90mins)].values.tolist()

        # under 60 minutes
        nStopsFrV_U60 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs60mins)].index) / denominator
        nStopsToV_U60 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs60mins)].index) / denominator
        stopsFrV_U60 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs60mins)].values.tolist()
        stopsToV_U60 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs60mins)].values.tolist()

        # under 90 minutes
        nStopsFrV_U90 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs90mins)].index) / denominator
        nStopsToV_U90 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] <= nSecs90mins)].index) / denominator
        stopsFrV_U90 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs90mins)].values.tolist()
        stopsToV_U90 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] <= nSecs90mins)].values.tolist()

        # over 90 minutes
        nStopsFrV_O90 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] > nSecs90mins)].index) / denominator
        nStopsToV_O90 = len(dfAllShortestPaths.loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                   (dfAllShortestPaths['tripTime'] > nSecs90mins)].index) / denominator
        stopsFrV_O90 = dfAllShortestPaths['stopTo'].loc[(dfAllShortestPaths['stopFr'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] > nSecs90mins)].values.tolist()
        stopsToV_O90 = dfAllShortestPaths['stopFr'].loc[(dfAllShortestPaths['stopTo'] == stopv) &
                                                        (dfAllShortestPaths['tripTime'] > nSecs90mins)].values.tolist()

        proxDens_U30.append([stopv, nStopsFrV_U30, nStopsToV_U30, stopsFrV_U30, stopsToV_U30])
        proxDens_3060.append([stopv, nStopsFrV_3060, nStopsToV_3060, stopsFrV_3060, stopsToV_3060])
        proxDens_6090.append([stopv, nStopsFrV_6090, nStopsToV_6090, stopsFrV_6090, stopsToV_6090])
        proxDens_U60.append([stopv, nStopsFrV_U60, nStopsToV_U60, stopsFrV_U60, stopsToV_U60])
        proxDens_U90.append([stopv, nStopsFrV_U90, nStopsToV_U90, stopsFrV_U90, stopsToV_U90])
        proxDens_O90.append([stopv, nStopsFrV_O90, nStopsToV_O90, stopsFrV_O90, stopsToV_O90])

    dfProxDensU30 = pd.DataFrame(proxDens_U30,
                                 columns = ['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])
    dfProxDens3060 = pd.DataFrame(proxDens_3060,
                                  columns=['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])
    dfProxDens6090 = pd.DataFrame(proxDens_6090,
                                  columns=['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])
    dfProxDensU60 = pd.DataFrame(proxDens_U60,
                                 columns=['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])
    dfProxDensU90 = pd.DataFrame(proxDens_U90,
                                 columns=['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])
    dfProxDensO90 = pd.DataFrame(proxDens_O90,
                                 columns=['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV'])

    return dfProxDensU30, dfProxDens3060, dfProxDens6090, dfProxDensU60, dfProxDensU90, dfProxDensO90

# ======================================================================================================================
def mkDfStopAttribs(G, dfBetwCentral, dfProxDensU30, dfProxDens3060, dfProxDens6090,
                    dfProxDensU60, dfProxDensU90, dfProxDensO90):
    '''
    columns in input dataframes are ['stopv', 'normProxDensFrV', 'normProxDensToV', 'stopsFrV', 'stopsToV']
    '''
    # change column names
    dfBetwCentral = dfBetwCentral.rename(columns={'stop': 'StationId',
                                                  'normBetwCentral': const.GNodeAttribs.cbTime.name})
    dfProxDensU30 = dfProxDensU30.rename(columns={'stopv': 'StationId',
                                                  'normProxDensFrV': 'frV_u30', 'normProxDensToV': 'toV_u30',
                                                  'stopsFrV': 'ndfrV_u30', 'stopsToV': 'ndtoV_u30'})
    dfProxDens3060 = dfProxDens3060.rename(columns={'stopv': 'StationId',
                                                    'normProxDensFrV': 'frV_3060', 'normProxDensToV': 'toV_3060',
                                                    'stopsFrV': 'ndfrV_3060', 'stopsToV': 'ndtoV_3060'})
    dfProxDens6090 = dfProxDens6090.rename(columns={'stopv': 'StationId',
                                                    'normProxDensFrV': 'frV_6090', 'normProxDensToV': 'toV_6090',
                                                    'stopsFrV': 'ndfrV_6090', 'stopsToV': 'ndtoV_6090'})
    dfProxDensU60 = dfProxDensU60.rename(columns={'stopv': 'StationId',
                                                  'normProxDensFrV': 'frV_u60', 'normProxDensToV': 'toV_u60',
                                                  'stopsFrV': 'ndfrV_u60', 'stopsToV': 'ndtoV_u60'})
    dfProxDensU90 = dfProxDensU90.rename(columns={'stopv': 'StationId',
                                                  'normProxDensFrV': 'frV_u90', 'normProxDensToV': 'toV_u90',
                                                  'stopsFrV': 'ndfrV_u90', 'stopsToV': 'ndtoV_u90'})
    dfProxDensO90 = dfProxDensO90.rename(columns={'stopv': 'StationId',
                                                  'normProxDensFrV': 'frV_o90', 'normProxDensToV': 'toV_o90',
                                                  'stopsFrV': 'ndfrV_o90', 'stopsToV': 'ndtoV_o90'})

    latCentroid, lngCentroid = utils.calcCentroidOfStops(G)
    geoAttribs = []
    for node in G.nodes:
        lng = G.nodes[node][const.GNodeAttribs.Lng.name]
        lat = G.nodes[node][const.GNodeAttribs.Lat.name]
        distFromCentroid = utils.calcGeodesicDist([lng, lat], [lngCentroid, latCentroid]) / 1000  # in km
        geoAttribs.append([node, lat, lng, distFromCentroid])
    dfGeoAttribs = pd.DataFrame(geoAttribs, columns=['StationId', 'Lat', 'Lng', 'R'])
    dfGeoAttribs['range'] = dfGeoAttribs['R'].apply(utils.getRange)

    dfStopAttribs = pd.merge(dfProxDensU30, dfProxDens3060, how='left', left_on=['StationId'], right_on=['StationId'])
    dfStopAttribs = pd.merge(dfStopAttribs, dfProxDens6090, how='left', left_on=['StationId'], right_on=['StationId'])
    dfStopAttribs = pd.merge(dfStopAttribs, dfProxDensU60, how='left', left_on=['StationId'], right_on=['StationId'])
    dfStopAttribs = pd.merge(dfStopAttribs, dfProxDensU90, how='left', left_on=['StationId'], right_on=['StationId'])
    dfStopAttribs = pd.merge(dfStopAttribs, dfProxDensO90, how='left', left_on=['StationId'], right_on=['StationId'])
    # merges geoAttribs
    dfStopAttribs = pd.merge(dfStopAttribs, dfGeoAttribs, how='left', left_on=['StationId'], right_on=['StationId'])
    # merge betwCentral
    dfStopAttribs = pd.merge(dfStopAttribs, dfBetwCentral, how='left', left_on=['StationId'], right_on=['StationId'])

    return dfStopAttribs

# ======================================================================================================================
def mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, proxDensAsOrigin, proxDensAsDest, yAxisTitle, filename):
    dfSns = pd.DataFrame()

    dfTmp = dfStopAttribs[[proxDensAsOrigin, 'range']]
    dfTmp = dfTmp.assign(type='asOrigin')
    dfTmp = dfTmp.rename(columns={proxDensAsOrigin: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    dfTmp = dfStopAttribs[[proxDensAsDest, 'range']]
    dfTmp = dfTmp.assign(type='asDestination')
    dfTmp = dfTmp.rename(columns={proxDensAsDest: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    fig = plt.figure(figsize=(5,5))

    sns.set(style="whitegrid")
    ax = sns.violinplot(x="range", y=yAxisTitle, hue='type', data=dfSns,
                        palette='muted', order=rangeCats, split=True, inner='quartile', cut=0, scale='count')
    # ax.set_xticklabels(rangeCats, rotation=90)
    ax.set_ylim([0, 1])
    ax.yaxis.grid(True)
    fig.savefig(filename, bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def mkViolinPlotsProxDens_ttgVsTopo(ttgDfStopAttribs, topoDfStopAttribs, rangeCats, proxDens, yAxisTitle, filename):
    dfSns = pd.DataFrame()

    dfTmp = ttgDfStopAttribs[[proxDens, 'range']]
    dfTmp = dfTmp.assign(type = 'from temporal topological analysis')
    dfTmp = dfTmp.rename(columns = {proxDens: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    dfTmp = topoDfStopAttribs[[proxDens, 'range']]
    dfTmp = dfTmp.assign(type='from topological analysis')
    dfTmp = dfTmp.rename(columns={proxDens: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    fig = plt.figure(figsize=(15, 10))

    sns.set(style="whitegrid")
    ax = sns.violinplot(x="range", y=yAxisTitle, hue='type', data=dfSns,
                        palette='muted', order=rangeCats, split=True, inner='quartile', cut=0, scale='count')
    # ax.set_xticklabels(rangeCats, rotation=90)
    ax.set_ylim([0, 1])
    ax.yaxis.grid(True)
    fig.savefig(filename, bbox_inches='tight', pad_inches=1)


# ======================================================================================================================
def getPathDirRouteIDs(pathNodes, ndIDict):
    return list(set(['%s_%s' % (ndIDict[node].split('_')[0], ndIDict[node].split('_')[1]) for node in pathNodes]))

def getPathNodeDetail(pathNodes, ndIDict):
    return [ndIDict[node] for node in pathNodes]

def collectRouteIDs(stopFrs, allPathsDetail, processID, outputFolder):
    dfPaths_stopFrs = allPathsDetail.loc[allPathsDetail['stopFr'].isin(stopFrs)]
    routeIDs_stopFrs = pd.Series()
    counts = 0
    starttime = time.perf_counter()
    for idx,row in dfPaths_stopFrs.iterrows():
        routeIDs_stopFrs = pd.concat([routeIDs_stopFrs, pd.Series(row['routeIDs'])])
        counts += 1
        if counts % 1e5 == 0:
            print('%d/%d completed, took %.4g mins' %  (counts, dfPaths_stopFrs.shape[0],
                                                        (time.perf_counter() - starttime)/60))

    with open('%s/routeIDs_stopFrs_%d.pkl' % (outputFolder, processID), 'wb') as f:
        pickle.dump(routeIDs_stopFrs, f)

# ======================================================================================================================
def getRouteIDs(shortestPathFiles, ndIDict):
    # consolidates shortest path results from multiprocessing
    allPathsDetail = pd.DataFrame()
    for file in shortestPathFiles:
        print('reading file %s... ' % file)
        shortestPaths = pickle.load(open(file, 'rb'))
        dfShortestPaths = pd.DataFrame(shortestPaths,
                                       columns=['stopFr', 'stopTo', 'timeFr', 'timeTo', 'path', 'nTransfers'])
        dfShortestPaths['routeIDs'] = dfShortestPaths.apply(lambda row: getPathDirRouteIDs(row['path'], ndIDict),
                                                            axis=1)
        dfShortestPaths['pathNodeDetail'] = dfShortestPaths.apply(lambda row: getPathNodeDetail(row['path'], ndIDict),
                                                                  axis=1)
        allPathsDetail = pd.concat([allPathsDetail,
                                    dfShortestPaths[['stopFr', 'stopTo', 'routeIDs', 'pathNodeDetail']]])

    # concate all routeIDs allPathsDetail['routeIDs] into a list
    starttime = time.perf_counter()
    allRouteIDList = dfShortestPaths['routeIDs'].sum()
    print('allRouteIDList completed took %.4g seconds' % (time.perf_counter() - starttime))
    # calculates frequency of each routeID and normalises these counts
    nStops = len(allPathsDetail['stopFr'].unique().tolist())
    dfAllRouteIDs = pd.DataFrame({'routeID':allRouteIDList})
    routeGrps = dfAllRouteIDs.groupby('routeID')['routeID'].count()
    dfRouteGrps = routeGrps.to_frame()
    dfRouteGrps = dfRouteGrps.rename(columns={'routeID': 'counts'})
    normFactor = (nStops - 1) * (nStops - 2)
    dfRouteGrps['fraction'] = dfRouteGrps['counts'] / normFactor

    return allPathsDetail, dfRouteGrps


# ======================================================================================================================
def runStuffOnBUDF(G, gttShortestPathsFolder, nProcesses, ndIDict, startHrStr, endHrStr):
    outputFolder = '%s/%s_%s' % (const.ttgAnalysisFolder, startHrStr, endHrStr)
    if os.path.exists(outputFolder) == False:
        os.mkdir(outputFolder)

    # CONSOLIDATES SHORTEST PATH RESULTS FROM MULTIPROCESSING - RUN ON BUDF
    shortestPathsFiles = ['%s/shortestPaths_%d.pkl' % (gttShortestPathsFolder, i) for i in range(nProcesses)]
    dfAllShortestPaths = consolidateOutputFiles(shortestPathsFiles, dfAllTransfers=None, ndIDict=ndIDict)
    with open('%s/%s_%s/dfAllShortestPaths.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfAllShortestPaths, f)

    dfAllPathsDetail, dfRouteGrps = getRouteIDs(shortestPathsFiles, ndIDict)
    with open('%s/dfAllPathsDetail.pkl' % outputFolder, 'wb') as f:
        pickle.dump(dfAllPathsDetail, f)
    with open('%s/dfRouteGrps.pkl' % outputFolder, 'wb') as f:
        pickle.dump(dfRouteGrps, f)

    # CALCULATES PROXIMITY DENSITY AND DUMP RESULTS TO PICKLE FILES - RUN ON BUDF
    dfProxDensU30, dfProxDens3060, dfProxDens6090, dfProxDensU60, dfProxDensU90, dfProxDensO90 = \
        calcProxDensity(allStops=list(G.nodes), dfAllShortestPaths=dfAllShortestPaths)
    with open('%s/%s_%s/dfProxDensU30.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDensU30, f)
    with open('%s/%s_%s/dfProxDens3060.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDens3060, f)
    with open('%s/%s_%s/dfProxDens6090.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDens6090, f)
    with open('%s/%s_%s/dfProxDensU60.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDensU60, f)
    with open('%s/%s_%s/dfProxDensU90.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDensU90, f)
    with open('%s/%s_%s/dfProxDensO90.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'wb') as f:
        pickle.dump(dfProxDensO90, f)

    # CALCULATES BETWEENNESS CENTRALITY - RUN ON BUDF
    calcBetwCentralMultip(allStops=list(G.nodes), nProcesses=4, dfAllShortestPaths=dfAllShortestPaths,
                          outputFolder='%s/%s_%s' % (const.ttgAnalysisFolder, startHrStr, endHrStr))
    # TODO there should be a check here to make sure that output file from all processes are available before
    #  consolidating the results
    consolidateBetwCentrals(nProcesses=4, outputFolder='%s/%s_%s' % (const.ttgAnalysisFolder, startHrStr, endHrStr))

# ======================================================================================================================
def mkDfStopAttribs_ttg(G_wNdAttribs, startHrStr, endHrStr):
    dfProxDensU30 = pickle.load(
        open('%s/%s_%s/dfProxDensU30.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfProxDens3060 = pickle.load(
        open('%s/%s_%s/dfProxDens3060.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfProxDens6090 = pickle.load(
        open('%s/%s_%s/dfProxDens6090.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfProxDensU60 = pickle.load(
        open('%s/%s_%s/dfProxDensU60.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfProxDensU90 = pickle.load(
        open('%s/%s_%s/dfProxDensU90.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfProxDensO90 = pickle.load(
        open('%s/%s_%s/dfProxDensO90.pkl' % (const.ttgAnalysisFolder, startHrStr, endHrStr), 'rb'))
    dfBetwCentral = pd.read_csv('%s/%s_%s/ttgBC_All.csv' % (const.ttgAnalysisFolder, startHrStr, endHrStr),
                                delimiter=',')
    dfStopAttribs = mkDfStopAttribs(G_wNdAttribs, dfBetwCentral,
                                    dfProxDensU30, dfProxDens3060, dfProxDens6090,
                                    dfProxDensU60, dfProxDensU90, dfProxDensO90)
    return dfStopAttribs

# ======================================================================================================================
def mkViolinPltProxDensByTime(nAllStops, dfStopAttribs, ax, xTitle, yTitle=False):
    dfSns = pd.DataFrame()
    timeOrder = ['under 30 mins', '30-60 mins', '60-90 mins', 'over 90 mins']
    attribList = {'frV_u30': ['under 30 mins', 'Destination proximity density'], # 'as Origin'
                  'frV_3060': ['30-60 mins', 'Destination proximity density'], # 'as Origin'
                  'frV_6090': ['60-90 mins', 'Destination proximity density'], # 'as Origin'
                  'frV_o90': ['over 90 mins', 'Destination proximity density'], # 'as Origin'
                  'toV_u30': ['under 30 mins', 'Origin proximity density'], # 'as Destination'
                  'toV_3060': ['30-60 mins', 'Origin proximity density'], # 'as Destination'
                  'toV_6090': ['60-90 mins', 'Origin proximity density'], # 'as Destination'
                  'toV_o90': ['over 90 mins', 'Origin proximity density']} # 'as Destination'

    for attrib in attribList.keys():
        dfTmp = dfStopAttribs[[attrib]]
        dfTmp[attrib] = dfTmp[attrib] * (nAllStops - 1)  # uses the number of stops, not fraction
        dfTmp[attrib] = dfTmp[attrib].round(0).astype(int)
        print(dfTmp[attrib].describe())
        dfTmp = dfTmp.rename(columns={attrib: 'Number of stops'})
        dfTmp = dfTmp.assign(time=attribList[attrib][0])
        dfTmp = dfTmp.assign(type=attribList[attrib][1])
        dfSns = pd.concat([dfSns, dfTmp])

    sns.set(style='whitegrid')
    sns.violinplot(x='time', y='Number of stops', hue='type', data=dfSns, ax=ax,
                   order=timeOrder, split=True, inner='quartile', cut=0, scale='count')
    ax.legend().set_title('')
    ax.legend(fontsize=10)
    #plt.setp(ax.get_legend().get_texts(), fontsize='2')  # for legend text
    #plt.setp(ax.get_legend().get_title(), fontsize='2')  # for legend title
    ax.set_xticklabels(timeOrder, rotation=90)
    ax.set_xlabel(xTitle)
    if yTitle==False:
        ax.set_ylabel('')
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkBoxplotProxDensDiffByTime_OvsD(nAllStops, dfStopAttribs, ax, xTitle, yTitle=False):
    dfSns = pd.DataFrame()
    timeOrder = ['under 30 mins', '30-60 mins', '60-90 mins', 'over 90 mins']

    myDict = {'under 30 mins': ['frV_u30', 'toV_u30'],
              '30-60 mins': ['frV_3060', 'toV_3060'],
              '60-90 mins': ['frV_6090', 'toV_6090'],
              'over 90 mins': ['frV_o90', 'toV_o90'],}

    for time in myDict.keys():
        dfTmp = pd.DataFrame()
        dfTmp['absDiffProxDens'] = dfStopAttribs[myDict[time][0]] - dfStopAttribs[myDict[time][1]]
        #dfTmp['absDiffProxDens'] = abs(dfStopAttribs[myDict[time][0]] - dfStopAttribs[myDict[time][1]])
        dfTmp['absDiffProxDens'] = dfTmp['absDiffProxDens']*(nAllStops - 1)
        dfTmp['absDiffProxDens'] = dfTmp['absDiffProxDens'].round(0).astype(int)
        print(dfTmp['absDiffProxDens'].describe())
        dfTmp = dfTmp.rename(columns={'absDiffProxDens': 'Absolute difference of proximity density'})
        dfTmp = dfTmp.assign(time = time)
        dfSns = pd.concat([dfSns, dfTmp])

    sns.set(style='whitegrid')
    sns.boxplot(x='time', y='Absolute difference of proximity density', data=dfSns, palette='muted', order=timeOrder,
                ax=ax, fliersize=.5, color=mcolors.CSS4_COLORS['lightgrey'])
    ax.set_xticklabels(timeOrder, rotation=90)
    ax.set_xlabel(xTitle)
    if yTitle == False:
        ax.set_ylabel('')
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkBoxPlotProxDensDiffByTime_ttgVsTopo(nAllStops, dfStopAttribs, dfStopAttribsTopo, ax, xTitle, yTitle=False):
    dfSns = pd.DataFrame()
    timeOrder = ['under 30 mins', '30-60 mins', '60-90 mins', 'over 90 mins']

    myDict = {'d_frV_u30': ['frV_u30', 'frV_u30_topo', 'Destination proximity density', 'under 30 mins'], # 'as Origin'
              'd_frV_3060': ['frV_3060', 'frV_3060_topo', 'Destination proximity density', '30-60 mins'], # 'as Origin'
              'd_frV_6090': ['frV_6090', 'frV_6090_topo', 'Destination proximity density', '60-90 mins'], # 'as Origin'
              'd_frV_o90': ['frV_o90', 'frV_o90_topo', 'Destination proximity density', 'over 90 mins'], # 'as Origin'
              'd_toV_u30': ['toV_u30', 'toV_u30_topo', 'Origin proximity density', 'under 30 mins'], # 'as Destination'
              'd_toV_3060': ['toV_3060', 'toV_3060_topo', 'Origin proximity density', '30-60 mins'], # 'as Destination'
              'd_toV_6090': ['toV_6090', 'toV_6090_topo', 'Origin proximity density', '60-90 mins'], # 'as Destination'
              'd_toV_o90': ['toV_o90', 'toV_o90_topo', 'Origin proximity density', 'over 90 mins']} # 'as Destination'

    #dfStopAttribs = dfStopAttribs.sort_values(by=['StationId'])
    #dfStopAttribsTopo = dfStopAttribsTopo.sort_values(by=['StationId'])
    attribColumns = ['StationId', 'frV_u30', 'frV_3060', 'frV_6090', 'frV_o90',
                     'toV_u30', 'toV_3060', 'toV_6090', 'toV_o90']
    dfBoth = dfStopAttribsTopo[attribColumns]
    dfBoth = dfBoth.rename(columns={'frV_u30': 'frV_u30_topo'})
    dfBoth = dfBoth.rename(columns={'frV_3060': 'frV_3060_topo'})
    dfBoth = dfBoth.rename(columns={'frV_6090': 'frV_6090_topo'})
    dfBoth = dfBoth.rename(columns={'frV_o90': 'frV_o90_topo'})
    dfBoth = dfBoth.rename(columns={'toV_u30': 'toV_u30_topo'})
    dfBoth = dfBoth.rename(columns={'toV_3060': 'toV_3060_topo'})
    dfBoth = dfBoth.rename(columns={'toV_6090': 'toV_6090_topo'})
    dfBoth = dfBoth.rename(columns={'toV_o90': 'toV_o90_topo'})
    dfBoth = pd.merge(dfBoth, dfStopAttribs[attribColumns], how='left', left_on=['StationId'], right_on=['StationId'])

    for attrib in myDict.keys():
        dfTmp = pd.DataFrame()
        dfTmp[attrib] = dfBoth[myDict[attrib][0]] - dfBoth[myDict[attrib][1]]
        dfTmp[attrib] = dfTmp[attrib] * (nAllStops-1)
        dfTmp[attrib] = dfTmp[attrib].round(0).astype(int)
        print(dfTmp[attrib].describe())
        dfTmp = dfTmp.rename(columns={attrib: 'Number of stops'})
        dfTmp = dfTmp.assign(type = myDict[attrib][2])
        dfTmp = dfTmp.assign(time = myDict[attrib][3])
        dfSns = pd.concat([dfSns, dfTmp])

    sns.set(style='whitegrid')
    sns.boxplot(x='time', y='Number of stops', data=dfSns, hue='type',
                palette='muted', order=timeOrder, ax=ax, fliersize=.5, color=mcolors.CSS4_COLORS['lightgrey'])
    plt.setp(ax.get_legend().get_texts(), fontsize='10')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='10')  # for legend title
    ax.legend().set_title('')
    ax.set_xticklabels(timeOrder, rotation=90)
    ax.set_xlabel(xTitle)
    if yTitle == False:
        ax.set_ylabel('')
    ax.set_ylim(-2000,2000)
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkViolinPlotAvgDistByTime(dfStopAttribs, ax, xTitle, yTitle=False):
    dfSns = pd.DataFrame()
    timeOrder = ['under 30 mins', '30-60 mins', '60-90 mins', 'over 90 mins']

    myDict = {'avgDist_ndfrV_u30': ['Destination proximity distance', 'under 30 mins'],# 'as Origin'
              'avgDist_ndfrV_3060': ['Destination proximity distance', '30-60 mins'],# 'as Origin'
              'avgDist_ndfrV_6090': ['Destination proximity distance', '60-90 mins'],# 'as Origin'
              'avgDist_ndfrV_o90': ['Destination proximity distance', 'over 90 mins'],# 'as Origin'
              'avgDist_ndtoV_u30': ['Origin proximity distance', 'under 30 mins'],# 'as Destination'
              'avgDist_ndtoV_3060': ['Origin proximity distance', '30-60 mins'],# 'as Destination'
              'avgDist_ndtoV_6090': ['Origin proximity distance', '60-90 mins'],# 'as Destination'
              'avgDist_ndtoV_o90': ['Origin proximity distance', 'over 90 mins']}# 'as Destination'

    for attrib in myDict.keys():
        dfTmp = dfStopAttribs[[attrib]]
        print(dfStopAttribs[attrib].describe())
        dfTmp = dfTmp.rename(columns={attrib: 'Average distance (km)'})
        dfTmp = dfTmp.assign(type = myDict[attrib][0])
        dfTmp = dfTmp.assign(time = myDict[attrib][1])
        dfSns = pd.concat([dfSns, dfTmp])

    sns.set(style='whitegrid')
    sns.violinplot(x='time', y='Average distance (km)', hue='type', data=dfSns, ax=ax, order=timeOrder, split=True,
                   inner='quartile', cut=0, scale='count')
    #plt.setp(ax.get_legend().get_texts(), fontsize='10')  # for legend text
    #plt.setp(ax.get_legend().get_title(), fontsize='10')  # for legend title
    ax.legend().set_title('')
    ax.legend(fontsize=10)
    ax.set_xticklabels(timeOrder, rotation=90)
    ax.set_xlabel(xTitle)
    if yTitle == False:
        ax.set_ylabel('')
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkBoxPlotAvgDistDiffByTime_ttgVsTopo(dfStopAttribs_ttg, dfStopAttribs_topo, ax, xTitle, yTitle=False):
    dfSns = pd.DataFrame()
    timeOrder = ['under 30 mins', '30-60 mins', '60-90 mins', 'over 90 mins']
    myDict = {'d_avgDist_ndfrV_u30': ['avgDist_ndfrV_u30', 'avgDist_ndfrV_u30_topo',
                                      'Destination proximity distance','under 30 mins'], # 'as Origin'
              'd_avgDist_ndfrV_3060': ['avgDist_ndfrV_3060', 'avgDist_ndfrV_3060_topo',
                                       'Destination proximity distance', '30-60 mins'],# 'as Origin'
              'd_avgDist_ndfrV_6090': ['avgDist_ndfrV_6090', 'avgDist_ndfrV_6090_topo',
                                       'Destination proximity distance', '60-90 mins'],# 'as Origin'
              'd_avgDist_ndfrV_o90': ['avgDist_ndfrV_o90', 'avgDist_ndfrV_o90_topo',
                                      'Destination proximity distance', 'over 90 mins'],# 'as Origin'
              'd_avgDist_ndtoV_u30': ['avgDist_ndtoV_u30', 'avgDist_ndtoV_u30_topo',
                                      'Origin proximity distance', 'under 30 mins'],# 'as Destination'
              'd_avgDist_ndtoV_3060': ['avgDist_ndtoV_3060', 'avgDist_ndtoV_3060_topo',
                                       'Origin proximity distance', '30-60 mins'],# 'as Destination'
              'd_avgDist_ndtoV_6090': ['avgDist_ndtoV_6090', 'avgDist_ndtoV_6090_topo',
                                       'Origin proximity distance', '60-90 mins'],# 'as Destination'
              'd_avgDist_ndtoV_o90': ['avgDist_ndtoV_o90', 'avgDist_ndtoV_o90_topo',
                                      'Origin proximity distance', 'over 90 mins']}# 'as Destination'

    attribColumns = ['StationId', 'avgDist_ndfrV_u30', 'avgDist_ndfrV_3060', 'avgDist_ndfrV_6090', 'avgDist_ndfrV_o90',
                     'avgDist_ndtoV_u30', 'avgDist_ndtoV_3060', 'avgDist_ndtoV_6090', 'avgDist_ndtoV_o90']
    dfBoth = dfStopAttribs_topo[attribColumns]
    dfBoth = dfBoth.rename(columns = {'avgDist_ndfrV_u30': 'avgDist_ndfrV_u30_topo',
                                      'avgDist_ndfrV_3060': 'avgDist_ndfrV_3060_topo',
                                      'avgDist_ndfrV_6090': 'avgDist_ndfrV_6090_topo',
                                      'avgDist_ndfrV_o90': 'avgDist_ndfrV_o90_topo',
                                      'avgDist_ndtoV_u30': 'avgDist_ndtoV_u30_topo',
                                      'avgDist_ndtoV_3060': 'avgDist_ndtoV_3060_topo',
                                      'avgDist_ndtoV_6090': 'avgDist_ndtoV_6090_topo',
                                      'avgDist_ndtoV_o90': 'avgDist_ndtoV_o90_topo'})
    dfBoth = pd.merge(dfBoth, dfStopAttribs_ttg[attribColumns], how='left', left_on=['StationId'],
                      right_on=['StationId'])

    for attrib in myDict.keys():
        dfTmp = pd.DataFrame()
        #stopAttrib = myDict[attrib][0]
        dfTmp[attrib] = dfBoth[myDict[attrib][0]] - dfBoth[myDict[attrib][1]]
        print(dfTmp[attrib].describe())
        dfTmp = dfTmp.rename(columns={attrib: 'Difference of average distance (km)'})
        dfTmp = dfTmp.assign(type=myDict[attrib][2])
        dfTmp = dfTmp.assign(time=myDict[attrib][3])
        dfSns = pd.concat([dfSns, dfTmp])

    sns.set(style='whitegrid')
    sns.boxplot(x='time', y='Difference of average distance (km)', data=dfSns, hue='type',
                palette='muted', order=timeOrder, ax=ax, fliersize=.5, color=mcolors.CSS4_COLORS['lightgrey'])
    plt.setp(ax.get_legend().get_texts(), fontsize='10')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='10')  # for legend title
    ax.legend().set_title('')
    ax.set_xticklabels(timeOrder, rotation=90)
    ax.set_xlabel(xTitle)
    if yTitle == False:
        ax.set_ylabel('')
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkAvgDistPlots_U10km(dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720, dfStopAttribs_topo, stopPairsDist):
    '''
    :return:
    '''
    avgDistCols = ['StationId', 'avgDist_ndfrV_u30', 'avgDist_ndfrV_3060', 'avgDist_ndfrV_6090', 'avgDist_ndfrV_o90',
                   'avgDist_ndtoV_u30', 'avgDist_ndtoV_3060', 'avgDist_ndtoV_6090', 'avgDist_ndtoV_o90']

    dfStopAttribs_0710_wDist_f = '%s/dfStopsAttribs_0710_wStopPairDist.pkl' % const.picklesFolder
    if os.path.isfile(dfStopAttribs_0710_wDist_f):
        dfStopAttribs_0710_wDist = pickle.load(open(dfStopAttribs_0710_wDist_f, 'rb'))
    else:
        print('calcDistanceDistrib started dfStopAttribs_0710')
        dfStopAttribs_0710_wDist = calcDistanceDistrib(dfStopAttribs_0710, stopPairsDist)
        with open(dfStopAttribs_0710_wDist_f, 'wb') as f:
            pickle.dump(dfStopAttribs_0710_wDist, f)
    dfStopAvgDist_0710 = dfStopAttribs_0710_wDist[avgDistCols].loc[dfStopAttribs_0710_wDist['range'] == 'under 10km']
    dfStopAttribs_0710_wDist = pd.DataFrame()
    print('df for 0710 loaded')

    dfStopAttribs_1215_wDist_f = '%s/dfStopsAttribs_1215_wStopPairDist.pkl' % const.picklesFolder
    if os.path.isfile(dfStopAttribs_1215_wDist_f):
        dfStopAttribs_1215_wDist = pickle.load(open(dfStopAttribs_1215_wDist_f,'rb'))
    else:
        print('calcDistanceDistrib started dfStopAttribs_1215')
        dfStopAttribs_1215_wDist = calcDistanceDistrib(dfStopAttribs_1215, stopPairsDist)
        with open(dfStopAttribs_1215_wDist_f, 'wb') as f:
            pickle.dump(dfStopAttribs_1215_wDist, f)
    dfStopAvgDist_1215 = dfStopAttribs_1215_wDist[avgDistCols].loc[dfStopAttribs_1215_wDist['range'] == 'under 10km']
    dfStopAttribs_1215_wDist = pd.DataFrame()
    print('df for 1215 loaded')

    dfStopAttribs_1720_wDist_f = '%s/dfStopsAttribs_1720_wStopPairDist.pkl' % const.picklesFolder
    if os.path.isfile(dfStopAttribs_1720_wDist_f):
        dfStopAttribs_1720_wDist = pickle.load(open(dfStopAttribs_1720_wDist_f, 'rb'))
    else:
        print('calcDistanceDistrib started dfStopAttribs_1720')
        dfStopAttribs_1720_wDist = calcDistanceDistrib(dfStopAttribs_1720, stopPairsDist)
        with open(dfStopAttribs_1720_wDist_f, 'wb') as f:
            pickle.dump(dfStopAttribs_1720_wDist, f)
    dfStopAvgDist_1720 = dfStopAttribs_1720_wDist[avgDistCols].loc[dfStopAttribs_1720_wDist['range'] == 'under 10km']
    dfStopAttribs_1720_wDist = pd.DataFrame()
    print('df for 1720 loaded')

    dfStopAttribs_topo_wDist_f = '%s/dfStopsAttribs_topo_wStopPairDist.pkl' % const.picklesFolder
    if os.path.isfile(dfStopAttribs_topo_wDist_f):
        dfStopAttribs_topo_wDist = pickle.load(open(dfStopAttribs_topo_wDist_f, 'rb'))
    else:
        print('calcDistanceDistrib started dfStopAttribs_topo')
        dfStopAttribs_topo_wDist = calcDistanceDistrib(dfStopAttribs_topo, stopPairsDist)
        with open(dfStopAttribs_topo_wDist_f, 'wb') as f:
            pickle.dump(dfStopAttribs_topo_wDist, f)
    dfStopAvgDist_topo = dfStopAttribs_topo_wDist[avgDistCols].loc[dfStopAttribs_topo_wDist['range'] == 'under 10km']
    dfStopAttribs_topo_wDist = pd.DataFrame()
    print('df for topo loaded')


    # makes violin plots of the average distance between each stop as the origin and as the destination (within 10km of
    # the network centroid) and all other stops that are within different categories of total travel time,
    # namely under 30 mins, 30-60 mins, 60-90mins, and over 90mins. The plots were made for departure times 0700,
    # 1200, 1700, and for the topological network.
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 4), sharey=True)
    print('\nmkViolinPlotAvgDistByTime departure time 0700')
    mkViolinPlotAvgDistByTime(dfStopAvgDist_0710, ax=axs[0], xTitle='(a) departure time 0700', yTitle=True)
    print('\nmkViolinPlotAvgDistByTime departure time 1215')
    mkViolinPlotAvgDistByTime(dfStopAvgDist_1215, ax=axs[1], xTitle='(b) departure time 1200')
    print('\nmkViolinPlotAvgDistByTime departure time 1720')
    mkViolinPlotAvgDistByTime(dfStopAvgDist_1720, ax=axs[2], xTitle='(c) departure time 1700')
    print('\nmkViolinPlotAvgDistByTime departure time topo')
    mkViolinPlotAvgDistByTime(dfStopAvgDist_topo, ax=axs[3], xTitle='(d) topological network')
    fig.savefig('%s/proxDensByTravTimes/violinPlots_avgDist_byDepTimes_U10km.png' % const.ttgAnalysisFolder)


    # makes boxplots of the difference between avg distance calculated at each departure hour vs avg distance
    # calculated by the topological model
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4), sharey=True)
    print('\nmkBoxPlotAvgDistDiffByTime_ttgVsTopo departure time 0700')
    mkBoxPlotAvgDistDiffByTime_ttgVsTopo(dfStopAvgDist_0710, dfStopAvgDist_topo, axs[0],
                                         xTitle='(a) departure time 0700 vs topological', yTitle=True)
    print('\nmkBoxPlotAvgDistDiffByTime_ttgVsTopo departure time 1215')
    mkBoxPlotAvgDistDiffByTime_ttgVsTopo(dfStopAvgDist_1215, dfStopAvgDist_topo, axs[1],
                                         xTitle='(b) departure time 1200 vs topological')
    print('\nmkBoxPlotAvgDistDiffByTime_ttgVsTopo departure time 1720')
    mkBoxPlotAvgDistDiffByTime_ttgVsTopo(dfStopAvgDist_1720, dfStopAvgDist_topo, axs[2],
                                         xTitle='(c) departure time 1700 vs topological')
    fig.savefig('%s/proxDensByTravTimes/boxPlots_avgDistDiff_byDepTimes_U10km.png' % const.ttgAnalysisFolder)


# ======================================================================================================================
def mkProxDensPlots_U10km(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720, dfStopAttribs_topo):
    nAllStops = len(list(G_wNdAttribs.nodes))

    dfStopAttribs_0710_u10km = dfStopAttribs_0710.loc[dfStopAttribs_0710['range'] == 'under 10km']
    dfStopAttribs_1215_u10km = dfStopAttribs_1215.loc[dfStopAttribs_1215['range'] == 'under 10km']
    dfStopAttribs_1720_u10km = dfStopAttribs_1720.loc[dfStopAttribs_1720['range'] == 'under 10km']
    dfStopAttribs_topo_u10km = dfStopAttribs_topo.loc[dfStopAttribs_topo['range'] == 'under 10km']

    '''
    columns = ['StationId', 'R', 'Lat', 'Lng', 'toV_u30', 'frV_u30', 'toV_3060', 'frV_3060', 'toV_6090', 'frV_6090',
               'toV_o90', 'frV_o90']
    dfStopAttribs_0710_u10km[columns].to_csv('%s/proxDensByTravTimes/dfStopAttribs_0710_u10km.csv' %
                                             const.ttgAnalysisFolder)
    dfStopAttribs_topo_u10km[columns].to_csv('%s/proxDensByTravTimes/dfStopAttribs_topo_u10km.csv' %
                                             const.ttgAnalysisFolder)
    '''

    # make violin plots of proximity density of nodes within 10km of the network centroid calculated as origin
    # and as destination from the temporal network model for 3 departure times, namely 7am, 12pm, 5pm and from the
    # topological network model.
    fig,axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12,4), sharey=True)
    print('\nmkViolinPltProxDensByTime - departure time 0700')
    mkViolinPltProxDensByTime(nAllStops, dfStopAttribs_0710_u10km, axs[0], xTitle = '(a) depature time 0700',
                              yTitle=True)
    print('\nmkViolinPltProxDensByTime - departure time 1200')
    mkViolinPltProxDensByTime(nAllStops, dfStopAttribs_1215_u10km, axs[1], xTitle = '(b) departure time 1200')
    print('\nmkViolinPltProxDensByTime - departure time 1700')
    mkViolinPltProxDensByTime(nAllStops, dfStopAttribs_1720_u10km, axs[2], xTitle = '(c) departure time 1700')
    print('\nmkViolinPltProxDensByTime - topo network')
    mkViolinPltProxDensByTime(nAllStops, dfStopAttribs_topo_u10km, axs[3], xTitle = '(d) topological network')
    fig.savefig('%s/proxDensByTravTimes/violinPlotsbyDepTimes_U10km.png' % const.ttgAnalysisFolder)
    print('saved file %s/proxDensByTravTimes/violinPlotsbyDepTimes_U10km.png' % const.ttgAnalysisFolder)

    '''
    # make boxplot of difference between the proximity density of stops within 10km of the network centroid
    # calculated as origin versus as destination, using the results from the temporal network model for 3 departure
    # times, namely 7am, 12pm, 5pm and from the results of the topological network model.
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 4), sharey=True)
    print('\nmkBoxplotProxDensDiffByTime_OvsD - depature time 0700')
    mkBoxplotProxDensDiffByTime_OvsD(nAllStops, dfStopAttribs_0710_u10km, axs[0], xTitle='depature time 0700',
                                     yTitle=True)
    print('\nmkBoxplotProxDensDiffByTime_OvsD - depature time 1200')
    mkBoxplotProxDensDiffByTime_OvsD(nAllStops, dfStopAttribs_1215_u10km, axs[1], xTitle='depature time 1200')
    print('\nmkBoxplotProxDensDiffByTime_OvsD - depature time 1700')
    mkBoxplotProxDensDiffByTime_OvsD(nAllStops, dfStopAttribs_1720_u10km, axs[2], xTitle='depature time 1700')
    print('\nmkBoxplotProxDensDiffByTime_OvsD - topo network')
    mkBoxplotProxDensDiffByTime_OvsD(nAllStops, dfStopAttribs_topo_u10km, axs[3], xTitle='topological network')
    fig.savefig('%s/proxDensByTravTimes/boxPlotsOvsD_byDepTimes_U10km.png' % const.ttgAnalysisFolder)
    '''


    # makes boxplot of the difference between proximity density at each stop calculated from the temporal model versus
    # that calculated from the topological model, as origin and as destination, for 3 departure times 0700, 1200, 1700.
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4), sharey=True)
    print('\nmkBoxPlotProxDensDiffByTime_ttgVsTopo - depature time 0700')
    mkBoxPlotProxDensDiffByTime_ttgVsTopo(nAllStops, dfStopAttribs_0710_u10km, dfStopAttribs_topo_u10km, axs[0],
                                          xTitle = '(a) departure time 0700 vs topological', yTitle=True)
    print('\nmkBoxPlotProxDensDiffByTime_ttgVsTopo - depature time 1200')
    mkBoxPlotProxDensDiffByTime_ttgVsTopo(nAllStops, dfStopAttribs_1215_u10km, dfStopAttribs_topo_u10km, axs[1],
                                          xTitle='(b) departure time 1200 vs topological', yTitle=False)
    print('\nmkBoxPlotProxDensDiffByTime_ttgVsTopo - depature time 1700')
    mkBoxPlotProxDensDiffByTime_ttgVsTopo(nAllStops, dfStopAttribs_1720_u10km, dfStopAttribs_topo_u10km, axs[2],
                                          xTitle='(c) departure time 1700 vs topological', yTitle=False)
    fig.savefig('%s/proxDensByTravTimes/boxPlots_ttgVsTopo_byDepTimes_U10km.png' % const.ttgAnalysisFolder)
    print('saved file %s/proxDensByTravTimes/boxPlots_ttgVsTopo_byDepTimes_U10km.png' % const.ttgAnalysisFolder)

# ======================================================================================================================
def mkDfTmp(stop1, stop2List):
    stop1List = [stop1] * len(stop2List)
    dfTmp = pd.DataFrame({'stop1': stop1List, 'stop2': stop2List})
    return dfTmp

def calcDistanceDistrib(dfNdAttribs, stopPairDist):
    #print('calcDistanceDistrib started')
    proxDensCats = ['ndfrV_u30', 'ndfrV_3060', 'ndfrV_6090', 'ndfrV_u60', 'ndfrV_u90', 'ndfrV_o90',
                    'ndtoV_u30', 'ndtoV_3060', 'ndtoV_6090', 'ndtoV_u60', 'ndtoV_u90', 'ndtoV_o90']
    starttime = time.perf_counter()
    for proxDensCat in proxDensCats:
        dfStopPairDist = pd.DataFrame()
        for idx, row in dfNdAttribs.iterrows():
            stop1 = row['StationId']
            stop2List = row[proxDensCat]#.tolist()
            dfStopPairDist = pd.concat([dfStopPairDist, mkDfTmp(stop1,stop2List)])
        dfStopPairDist['stopPair'] = dfStopPairDist.apply(lambda row: '%d_%d' % (row['stop1'],row['stop2']), axis=1)
        dfStopPairDist['stopPairDist'] = dfStopPairDist['stopPair'].map(stopPairDist)
        dfStopPairDist['stopPairDist'] = dfStopPairDist['stopPairDist'] * 1e-3 # converts to km
        stop1Groups = dfStopPairDist.groupby(['stop1'])['stopPairDist'].mean()
        dfNdAttribs = pd.merge(dfNdAttribs, stop1Groups.to_frame(), how='left', left_on=['StationId'], right_on=['stop1'])
        dfNdAttribs = dfNdAttribs.rename(columns = {'stopPairDist': 'avgDist_%s' % proxDensCat})
        print('\t%s completed - took %.4g mins' % (proxDensCat, (time.perf_counter() - starttime)/60))

    return dfNdAttribs

# ======================================================================================================================
def mkDistPlotBetwCentralByTime(nAllStops, dfStopAttribs, ax, xTitle, yTitle):
    '''
    :param nAllStops:
    :param dfStopAttribs:
    :param ax:
    :param xTitle:
    :param yTitle:
    :return:
    '''
    dfTmp = dfStopAttribs[['cbTime']]
    dfTmp['cbTime'] = dfTmp['cbTime'] * ((nAllStops-1)*(nAllStops-2))
    dfTmp['cbTime'] = dfTmp['cbTime'].round(0).astype(int)
    print(dfTmp['cbTime'].describe())
    sns.set(style='whitegrid')
    sns.distplot(dfTmp['cbTime'], kde=True, color='darkblue', ax=ax)
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    ax.yaxis.grid(True)

# ======================================================================================================================
def mkBetwCentralPlots_U10km(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720,
                             dfStopAttribs_topo):
    nAllStops = len(list(G_wNdAttribs.nodes))

    dfStopAttribs_0710_u10km = dfStopAttribs_0710.loc[dfStopAttribs_0710['range'] == 'under 10km']
    dfStopAttribs_1215_u10km = dfStopAttribs_1215.loc[dfStopAttribs_1215['range'] == 'under 10km']
    dfStopAttribs_1720_u10km = dfStopAttribs_1720.loc[dfStopAttribs_1720['range'] == 'under 10km']
    dfStopAttribs_topo_u10km = dfStopAttribs_topo.loc[dfStopAttribs_topo['range'] == 'under 10km']

    '''
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4), sharey=True)
    print('mkDistPlotBetwCentralByTime 0700')
    mkDistPlotBetwCentralByTime(nAllStops, dfStopAttribs_0710_u10km, axs[0], xTitle='depature time 0700', yTitle='')
    print('mkDistPlotBetwCentralByTime 1215')
    mkDistPlotBetwCentralByTime(nAllStops, dfStopAttribs_1215_u10km, axs[1], xTitle='depature time 1215', yTitle='')
    print('mkDistPlotBetwCentralByTime 1720')
    mkDistPlotBetwCentralByTime(nAllStops, dfStopAttribs_1720_u10km, axs[2], xTitle='depature time 1720', yTitle='')
    #print('mkDistPlotBetwCentralByTime topo')
    #mkDistPlotBetwCentralByTime(nAllStops, dfStopAttribs_topo_u10km, axs[3], xTitle='topological network', yTitle='')
    fig.savefig('%s/distPlots_betwCentral_byDepTimes_U10km.png' % const.ttgAnalysisFolder)
    '''

    fig = plt.figure(figsize=(12, 4))
    normFactor = (nAllStops-1)*(nAllStops-2)
    sns.set(style='whitegrid')
    sns.kdeplot(dfStopAttribs_0710_u10km['cbTime'] * normFactor, shade=False, color='r')
    sns.kdeplot(dfStopAttribs_0710_u10km['cbTime'] * normFactor, shade=False, color='g')
    sns.kdeplot(dfStopAttribs_0710_u10km['cbTime'] * normFactor, shade=False, color='b')
    fig.savefig('%s/kdePlots_betwCentral_byDepTimes_U10km.png' % const.ttgAnalysisFolder)

# ======================================================================================================================
def mkPlotsBetwCentral(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720,
                             dfStopAttribs_topo):
    nAllStops = len(list(G_wNdAttribs.nodes))
    normFactor = (nAllStops - 1) * (nAllStops - 2)

    dfBetwCentrals = dfStopAttribs_0710[['StationId', 'cbTime', 'Lat', 'Lng']]
    dfBetwCentrals = dfBetwCentrals.rename(columns={'cbTime': 'nPairs0710'})
    dfBetwCentrals = pd.merge(dfBetwCentrals, dfStopAttribs_1215[['StationId', 'cbTime']], how='left',
                              left_on=['StationId'], right_on=['StationId'])
    dfBetwCentrals = dfBetwCentrals.rename(columns={'cbTime': 'nPairs1215'})
    dfBetwCentrals = pd.merge(dfBetwCentrals, dfStopAttribs_1720[['StationId', 'cbTime']], how='left',
                              left_on=['StationId'], right_on=['StationId'])
    dfBetwCentrals = dfBetwCentrals.rename(columns={'cbTime': 'nPairs1720'})
    dfBetwCentrals = pd.merge(dfBetwCentrals, dfStopAttribs_topo[['StationId', 'cbTime']], how='left',
                              left_on=['StationId'], right_on=['StationId'])
    dfBetwCentrals = dfBetwCentrals.rename(columns={'cbTime': 'nPairstopo'})


    minVal = min(dfBetwCentrals['nPairs0710'].min(), dfBetwCentrals['nPairs1215'].min(),
                 dfBetwCentrals['nPairs1720'].min(), dfBetwCentrals['nPairstopo'].min())
    maxVal = max(dfBetwCentrals['nPairs0710'].max(), dfBetwCentrals['nPairs1215'].max(),
                 dfBetwCentrals['nPairs1720'].max(), dfBetwCentrals['nPairstopo'].max())
                 
    class cb(Enum):
        nPairs0710 = 'departure time 0700'
        nPairs1215 = 'departure time 1215'
        nPairs1720 = 'departure time 1720'
        nPairstopo = 'topological network'
    
    # produces map plots
    geoPlotter.plotBetweennessCentralities_v2(G_wNdAttribs, dfBetwCentrals,
                                              [cb.nPairs0710,cb.nPairs1215,cb.nPairs1720,cb.nPairstopo], minVal, maxVal,
                                              geoPlotter.makeBaseMap(), '%s/stopPairsAsBC.html' %
                                              const.ttgAnalysisFolder)

    '''
    # produces box plots of difference of number of stop pairs ttg vs topo
    dfBetwCentrals['nPairs0710'] = dfBetwCentrals['nPairs0710'] * normFactor
    dfBetwCentrals['nPairs0710'].round(0).astype(int)
    dfBetwCentrals['nPairs1215'] = dfBetwCentrals['nPairs1215'] * normFactor
    dfBetwCentrals['nPairs1215'].round(0).astype(int)
    dfBetwCentrals['nPairs1720'] = dfBetwCentrals['nPairs1720'] * normFactor
    dfBetwCentrals['nPairs1720'].round(0).astype(int)
    dfBetwCentrals['nPairstopo'] = dfBetwCentrals['nPairstopo'] * normFactor
    dfBetwCentrals['nPairstopo'].round(0).astype(int)

    myDict = {'d_nPairs0710': ['nPairs0710', 'departure time 0700'],
              'd_nPairs1215': ['nPairs1215', 'departure time 1200'],
              'd_nPairs1720': ['nPairs1720', 'departure time 1700']}

    dfSns = pd.DataFrame()
    for attrib in myDict.keys():
        dfTmp = pd.DataFrame()
        dfTmp[attrib] = dfBetwCentrals[myDict[attrib][0]] - dfBetwCentrals['nPairstopo']
        print(dfTmp[attrib].describe())
        dfTmp = dfTmp.rename(columns={attrib: 'Number of stop pairs'})
        dfTmp = dfTmp.assign(departTime=myDict[attrib][1])
        dfSns = pd.concat([dfSns,dfTmp])
    #dfBetwCentrals.to_csv('dfBetwCentrals.csv', index=False)
    #dfSns.to_csv('dfSns.csv', index=False)

    plotOrder = ['departure time 0700', 'departure time 1200', 'departure time 1700']
    fig = plt.figure()
    sns.set(style='whitegrid')
    ax = sns.boxplot(x='departTime', y='Number of stop pairs', data=dfSns, order=plotOrder,
                     fliersize=.5, color=mcolors.CSS4_COLORS['lightgray'])
    ax.set_xticklabels(plotOrder, rotation=0)
    ax.set_xlabel('')
    ax.yaxis.grid(True)
    ax.set_ylim(-.5e6, .5e6)
    ax.set_ylabel('Number of stop pairs')
    fig.savefig('%s/d_nStopPairsAsBC_ttgVstopo.png' % const.ttgAnalysisFolder)
    '''


# ======================================================================================================================
def main_ttgPostprocMultip():
    '''
    :return:
    '''

    # LOAD NECESSARY THINGS FROM PICKLES -------------------------------------------------------------------------------
    #gttTransit = pickle.load(open('%s/GttTransit_v2.pkl' % const.picklesFolder, 'rb'))
    #dfNdDetails = pickle.load(open('%s/dfNdDetails_v2.pkl' % const.picklesFolder, 'rb'))
    #ndIDictInv = pickle.load(open('%s/ndIDictInv_v2.pkl' % const.picklesFolder, 'rb'))
    #dfAllTransfers = pickle.load(open('%s/dfAllTransfers.pkl' % const.picklesFolder, 'rb'))
    stopPairsDist = pickle.load(open('%s/stopPairsDist.pkl' % const.picklesFolder, 'rb'))
    #ndIDict = pickle.load(open('%s/ndIDict_v2.pkl' % const.picklesFolder, 'rb'))
    G_wNdAttribs = pickle.load(open('%s/G_wNdAttribs.pkl' % const.picklesFolder, 'rb'))
    dfRoutes = pickle.load(open('%s/dfRoutes.pkl' % const.picklesFolder, 'rb'))
    # ------------------------------------------------------------------------------------------------------------------

    '''
    # CALCULATES STUFF ON BUDF -----------------------------------------------------------------------------------------
    startHr = 5
    endHr = 8
    nProcesses = 56
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    gttShortestPathsFolder = '%s/gttShortestPaths/%s_%s' % (const.trashSite, startHrStr, endHrStr)
    
    runStuffOnBUDF(G_wNdAttribs, gttShortestPathsFolder, nProcesses, ndIDict, startHrStr, endHrStr)
    # ------------------------------------------------------------------------------------------------------------------
    '''

    '''
    # PLOTS MOST USED DIRECTED ROUTES AT DIFFERENT DEPARTURE TIMES -----------------------------------------------------
    nTopRoutes = 21
    mapFile = '%s/mostUsedRoutes_%d.html' % (const.ttgAnalysisFolder, nTopRoutes)

    def getMostUsedRoutes(filename):
        dfRouteGrps = pickle.load(open(filename, 'rb'))
        dfRouteGrps = dfRouteGrps.sort_values(by=['counts'], ascending=False)
        dfTopRoutes = dfRouteGrps[['counts']].head(nTopRoutes)
        dfTopRoutes.reset_index(level=0, inplace=True)
        dfTopRoutes = dfTopRoutes.rename(columns={'routeID': 'dirRoute', 'counts': 'nPairs'})
        return dfTopRoutes

    dfTopRoutes_0710 = getMostUsedRoutes('%s/ttgAnalysis/0700_1000/dfRouteGrps.pkl' % const.outputsFolder)
    dfTopRoutes_1215 = getMostUsedRoutes('%s/ttgAnalysis/1200_1500/dfRouteGrps.pkl' % const.outputsFolder)
    dfTopRoutes_1720 = getMostUsedRoutes('%s/ttgAnalysis/1700_2000/dfRouteGrps.pkl' % const.outputsFolder)

    geoPlotter.plotTopRoutes(G_wNdAttribs, dfRoutes, geoPlotter.makeBaseMap(),
                               dfTopRoutes_0710, dfTopRoutes_1215, dfTopRoutes_1720, mapFile)
    # ------------------------------------------------------------------------------------------------------------------
    '''


    # PLOTS PROXIMITY DENSITY FOR MULTIPLE DEPARTURE TIMES AND FOR TOPOLOGICAL MODEL -----------------------------------
    startHr = 7
    endHr = 10
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    dfStopAttribs_0710 = mkDfStopAttribs_ttg(G_wNdAttribs, startHrStr, endHrStr)

    startHr = 12
    endHr = 15
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    dfStopAttribs_1215 = mkDfStopAttribs_ttg(G_wNdAttribs, startHrStr, endHrStr)

    startHr = 17
    endHr = 20
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    dfStopAttribs_1720 = mkDfStopAttribs_ttg(G_wNdAttribs, startHrStr, endHrStr)

    dfStopAttribs_topo = topoPostproc.mkDfNodeAttribs(G_wNdAttribs)

    mkProxDensPlots_U10km(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720, dfStopAttribs_topo)

    mkAvgDistPlots_U10km(dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720, dfStopAttribs_topo, stopPairsDist)

    #mkBetwCentralPlots_U10km(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720,
    #                         dfStopAttribs_topo)

    #mkPlotsBetwCentral(G_wNdAttribs, dfStopAttribs_0710, dfStopAttribs_1215, dfStopAttribs_1720,
    #                         dfStopAttribs_topo)


# ======================================================================================================================
def main_draft():
    '''
    :return:
    '''

    # LOAD NECESSARY THINGS FROM PICKLES -------------------------------------------------------------------------------
    # gttTransit = pickle.load(open('%s/GttTransit_v2.pkl' % const.picklesFolder, 'rb'))
    # dfNdDetails = pickle.load(open('%s/dfNdDetails_v2.pkl' % const.picklesFolder, 'rb'))
    # ndIDictInv = pickle.load(open('%s/ndIDictInv_v2.pkl' % const.picklesFolder, 'rb'))
    # dfAllTransfers = pickle.load(open('%s/dfAllTransfers.pkl' % const.picklesFolder, 'rb'))
    stopPairsDist = pickle.load(open('%s/stopPairsDist.pkl' % const.picklesFolder, 'rb'))
    # ndIDict = pickle.load(open('%s/ndIDict_v2.pkl' % const.picklesFolder, 'rb'))
    G_wNdAttribs = pickle.load(open('%s/G_wNdAttribs.pkl' % const.picklesFolder, 'rb'))
    # dfRoutes = pickle.load(open('%s/dfRoutes.pkl' % const.picklesFolder, 'rb'))
    # ------------------------------------------------------------------------------------------------------------------

    '''
    # CALCULATES STUFF ON BUDF -----------------------------------------------------------------------------------------
    startHr = 5
    endHr = 8
    nProcesses = 56
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    gttShortestPathsFolder = '%s/gttShortestPaths/%s_%s' % (const.trashSite, startHrStr, endHrStr)

    runStuffOnBUDF(G_wNdAttribs, gttShortestPathsFolder, nProcesses, ndIDict, startHrStr, endHrStr)
    # ------------------------------------------------------------------------------------------------------------------
    '''

    # SET PARAMETERS FOR PLOTTING INDIVIDUAL DEPARTURE TIME ------------------------------------------------------------
    startHr = 17
    endHr = 20
    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    #gttShortestPathsFolder = '%s/gttShortestPaths/%s_%s' % (const.trashSite, startHrStr, endHrStr)
    
    # PREPARE DF FOR ALL PLOTTING --------------------------------------------------------------------------------------
    # stop attributes from temporal network model
    dfStopAttribs = mkDfStopAttribs_ttg(G_wNdAttribs, startHrStr, endHrStr)
    # stop attributes from topological model
    topoDfStopAttribs = topoPostproc.mkDfNodeAttribs(G_wNdAttribs)
    # ------------------------------------------------------------------------------------------------------------------
    
    # MAKES MAP PLOT OF PROXIMITY DENSITIES ----------------------------------------------------------------------------
    mapPlotFolder = '%s/%s_%s/plots/mapPlots' % (const.ttgAnalysisFolder, startHrStr, endHrStr)
    geoPlotter.plotProximityDensity(G_wNdAttribs, dfStopAttribs, geoPlotter.makeBaseMap(),
                                    '%s/ttg_proxDensities.html' % mapPlotFolder)
    #geoPlotter.plotBetweennessCentralities(G_wNdAttribs, dfStopAttribs, const.GNodeAttribs.cbTime,
    #                                       geoPlotter.makeBaseMap(), '%s/ttg_betwCentralTime.html' % mapPlotFolder)
    # ------------------------------------------------------------------------------------------------------------------


    '''
    # MAKES VIOLIN PLOTS OF PROXIMITY DENSITY - FRV VS TOV -------------------------------------------------------------
    #rangeCats = ['under 5km', '5km - 10km', '10km - 15km', '15km - 20km', '20km - 30km', '30km - 40km', 'over 40km']
    rangeCats = ['under 10km', '10km - 20km', 'over 20km']
    violinPlotFolder = '%s/%s_%s/plots/violins/frV vs toV' % (const.ttgAnalysisFolder, startHrStr, endHrStr)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_u30', 'toV_u30',
                                 yAxisTitle = 'proximity density - under 30 minutes',
                                 filename = '%s/proxDens_u30.png' % violinPlotFolder)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_u60', 'toV_u60',
                                 yAxisTitle='proximity density - under 60 minutes',
                                 filename='%s/proxDens_u60.png' % violinPlotFolder)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_u90', 'toV_u90',
                                 yAxisTitle='proximity density - under 90 minutes',
                                 filename='%s/proxDens_u90.png' % violinPlotFolder)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_o90', 'toV_o90',
                                 yAxisTitle='proximity density - over 90 minutes',
                                 filename='%s/proxDens_o90.png' % violinPlotFolder)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_3060', 'toV_3060',
                                 yAxisTitle='proximity density - 30-60 minutes',
                                 filename='%s/proxDens_3060.png' % violinPlotFolder)
    mkViolinPlotsProxDens_FrVsTo(dfStopAttribs, rangeCats, 'frV_6090', 'toV_6090',
                                 yAxisTitle='proximity density - 60-90 minutes',
                                 filename='%s/proxDens_6090.png' % violinPlotFolder)
    # ------------------------------------------------------------------------------------------------------------------
    
    # MAKES VIOLIN PLOTS OF PROXIMITY DENSITY - (FRV TTG VS FRV TOPO) & (TOV TTG VS TOV TOPO) --------------------------
    rangeCats = ['under 10km', '10km - 20km', 'over 20km']
    violinPlotFolder = '%s/%s_%s/plots/violins/ttg vs topo/' % (const.ttgAnalysisFolder, startHrStr, endHrStr)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_u30',
                                    yAxisTitle='proximity density as Origin - under 30 minutes',
                                    filename='%s/proxDens_ttgTopo_fru30.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_u30',
                                    yAxisTitle='proximity density as Destination - under 30 minutes',
                                    filename='%s/proxDens_ttgTopo_tou30.png' % violinPlotFolder)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_3060',
                                    yAxisTitle='proximity density as Origin - 30-60 minutes',
                                    filename='%s/proxDens_ttgTopo_fr3060.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_3060',
                                    yAxisTitle='proximity density as Destination - 30-60 minutes',
                                    filename='%s/proxDens_ttgTopo_to3060.png' % violinPlotFolder)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_6090',
                                    yAxisTitle='proximity density as Origin - 60-90 minutes',
                                    filename='%s/proxDens_ttgTopo_fr6090.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_6090',
                                    yAxisTitle='proximity density as Destination - 60-90 minutes',
                                    filename='%s/proxDens_ttgTopo_to6090.png' % violinPlotFolder)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_u60',
                                    yAxisTitle='proximity density as Origin - under 60 minutes',
                                    filename='%s/proxDens_ttgTopo_fru60.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_u60',
                                    yAxisTitle='proximity density as Destination - under 60 minutes',
                                    filename='%s/proxDens_ttgTopo_tou60.png' % violinPlotFolder)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_u90',
                                    yAxisTitle='proximity density as Origin - under 90 minutes',
                                    filename='%s/proxDens_ttgTopo_fru90.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_u90',
                                    yAxisTitle='proximity density as Destination - under 90 minutes',
                                    filename='%s/proxDens_ttgTopo_tou90.png' % violinPlotFolder)

    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'frV_o90',
                                    yAxisTitle='proximity density as Origin - over 90 minutes',
                                    filename='%s/proxDens_ttgTopo_fro90.png' % violinPlotFolder)
    mkViolinPlotsProxDens_ttgVsTopo(dfStopAttribs, topoDfStopAttribs, rangeCats, 'toV_o90',
                                    yAxisTitle='proximity density as Destination - over 90 minutes',
                                    filename='%s/proxDens_ttgTopo_too90.png' % violinPlotFolder)
    # ------------------------------------------------------------------------------------------------------------------
    
    utils.pairedDifferenceTest(topoDfStopAttribs['toV_u30'].values.tolist(),
                               dfStopAttribs['toV_u30'].values.tolist())
    '''
