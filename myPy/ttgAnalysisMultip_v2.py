import pandas as pd
import pickle
import igraph
import time
import random
import multiprocessing
import os
import numpy as np

import constants as const

# ======================================================================================================================
def convert2Str(intList):
    strVals = ''
    for val in intList:
        strVals = '%s_%s' % (strVals, str(val))
    return strVals

# ======================================================================================================================
def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# ======================================================================================================================
def addTransferEdgesAllStops(gttTransit, dfAllTransfers, timeWindow):
    dfAllTransfers_timeFr = const.dfAllTransfersCols.timeFr.name
    dfAllTransfers_timeTo = const.dfAllTransfersCols.timeTo.name
    dfAllTransfers_nodeFr = const.dfAllTransfersCols.nodeFr.name
    dfAllTransfers_nodeTo = const.dfAllTransfersCols.nodeTo.name
    dfAllTransfers_type = const.dfAllTransfersCols.type.name
    wait = const.GttEdgeAttribs.typeWait.value
    walk = const.GttEdgeAttribs.typeWalk.value

    dfWait = dfAllTransfers[[dfAllTransfers_nodeFr, dfAllTransfers_nodeTo]].loc[
        (dfAllTransfers[dfAllTransfers_timeFr] >= timeWindow[0]) &
        (dfAllTransfers[dfAllTransfers_timeFr] <= timeWindow[1]) &
        (dfAllTransfers[dfAllTransfers_timeTo] >= timeWindow[0]) &
        (dfAllTransfers[dfAllTransfers_timeTo] <= timeWindow[1]) &
        (dfAllTransfers[dfAllTransfers_type] == wait)]
    dfWait['pair'] = dfWait.apply(lambda row: (row[dfAllTransfers_nodeFr], row[dfAllTransfers_nodeTo]), axis=1)
    waitEdgeList = dfWait['pair'].tolist()
    # adds waitEdgeList to gttTransit
    gttTransit.add_edges(waitEdgeList)
    # select those new edges, i.e. those with transfer being None
    waitEdgeSeq = gttTransit.es.select(transfer_in = [None])
    # assigns value 1 to all of these new edges
    waitEdgeSeq[const.GttEdgeAttribs.transfer.name] = 1
    waitEdgeSeq[const.GttEdgeAttribs.type.name] = wait

    dfWalk = dfAllTransfers[[dfAllTransfers_nodeFr, dfAllTransfers_nodeTo]].loc[
        (dfAllTransfers[dfAllTransfers_timeFr] >= timeWindow[0]) &
        (dfAllTransfers[dfAllTransfers_timeFr] <= timeWindow[1]) &
        (dfAllTransfers[dfAllTransfers_timeTo] >= timeWindow[0]) &
        (dfAllTransfers[dfAllTransfers_timeTo] <= timeWindow[1]) &
        (dfAllTransfers[dfAllTransfers_type] == walk)]
    dfWalk['pair'] = dfWalk.apply(lambda row: (row[dfAllTransfers_nodeFr], row[dfAllTransfers_nodeTo]), axis=1)
    walkEdgeList = dfWalk['pair'].tolist()
    # adds walkEdgeList to gttTransit
    gttTransit.add_edges(walkEdgeList)
    # selects those new edges, i.e. those with transfer being None
    walkEdgeSeq= gttTransit.es.select(transfer_in = [None])
    # assigns calue 1 to all of these new edges
    walkEdgeSeq[const.GttEdgeAttribs.transfer.name] = 1
    walkEdgeSeq[const.GttEdgeAttribs.type.name] = walk

    print('total wait edges %d' % len(waitEdgeList))
    print('total walk edges %d' % len(walkEdgeList))


# ======================================================================================================================
def extractTransferEdges(gttTransit, dfNdDetails, timeWindow):
    #gets subset of vertices within timeWindow
    dfNdDetails_ndId = const.dfNdDetailsCols.gttNdID.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name
    dfNdsSel = dfNdDetails[dfNdDetails_ndId].loc[(dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                                 (dfNdDetails[dfNdDetails_time] <= timeWindow[1])].values.tolist()

    # BE CAREFUL: id of vertices in the new subgraph are restarted and thus different to those in the original graph
    gttTransitSub = gttTransit.subgraph(vertices=dfNdsSel, implementation='auto')


    # checks whether transfer edges in the new sub graph match with those selected
    transferEdges = []
    walkEdgeSet = gttTransitSub.es.select(type_in=['walk'])
    for edge in walkEdgeSet:
        stopFr = int(edge.source_vertex['ndDesc'].split('_')[2])
        timeFr = int(edge.source_vertex['ndDesc'].split('_')[3])
        stopTo = int(edge.target_vertex['ndDesc'].split('_')[2])
        timeTo = int(edge.target_vertex['ndDesc'].split('_')[3])
        transferEdges.append([stopFr, timeFr, stopTo, timeTo, 'walk'])
    waitEdgeSet = gttTransitSub.es.select(type_in=['wait'])
    for edge in waitEdgeSet:
        stopFr = int(edge.source_vertex['ndDesc'].split('_')[2])
        timeFr = int(edge.source_vertex['ndDesc'].split('_')[3])
        stopTo = int(edge.target_vertex['ndDesc'].split('_')[2])
        timeTo = int(edge.target_vertex['ndDesc'].split('_')[3])
        transferEdges.append([stopFr, timeFr, stopTo, timeTo, 'wait'])
    dfTransferEdges = pd.DataFrame(transferEdges, columns = ['stopFr', 'timeFr', 'stopTo', 'timeTo', 'type'])
    #dfTransferEdges.to_csv('%s/dfTransfersSubGraph.csv' % const.trashSite, index=False)
    return dfTransferEdges

# ======================================================================================================================
def getMinTimeNodeEachDirRoute(dfStopFrNds):
    dfNdDetails_dirRoute = const.dfNdDetailsCols.dirRoute.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name
    dfNdDetails_ndID = const.dfNdDetailsCols.gttNdID.name

    groups = dfStopFrNds.groupby([dfNdDetails_dirRoute])
    nodesFr = []
    for name, group in groups:
        timeMin = group[dfNdDetails_time].min()
        ndTimeMin = group[dfNdDetails_ndID].loc[group[dfNdDetails_time] == timeMin].values[0]
        nodesFr.append([ndTimeMin, timeMin])

    if len(nodesFr)==0: return np.array(nodesFr) # notes: np.array(nodesFr).size == 0

    nodesFrNP = np.array(nodesFr)
    nodesFrNP = nodesFrNP[np.argsort(nodesFrNP[:, 1])]
    return nodesFrNP

def calcShortestPathsSubset_v3(gttTransit, stopFrList, allStops, timeWindow, dfNdDetails,
                            stopPairsDist, procNumber, gttShortestPathsFolder):
    # defines dfNdDetails columns
    dfNdDetails_stnID = const.dfNdDetailsCols.StationId.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name
    dfNdDetails_arrDep = const.dfNdDetailsCols.arrOrDep.name
    dfNdDetails_arr = const.dfNdDetailsCols.arr.name
    dfNdDetails_dep = const.dfNdDetailsCols.dep.name
    dfNdDetails_ndID = const.dfNdDetailsCols.gttNdID.name

    shortestPaths = []
    countStopFr = 0
    starttime = time.perf_counter()
    for stopFr in stopFrList:
        countStopTo = 0

        dfStopFrNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopFr) &
                                      (dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                      (dfNdDetails[dfNdDetails_time] <= timeWindow[1]) &
                                      (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_dep)]
        # dfStopFrNds = dfStopFrNds.sort_values(by = [dfNdDetails_time])
        nodesFrNP = getMinTimeNodeEachDirRoute(dfStopFrNds)

        if nodesFrNP.size == 0:
            print('WARNING: process %d - stopFr %d has no nodes' % (procNumber, stopFr))
            continue

        for stopTo in allStops:
            if stopFr == stopTo: continue

            dist = stopPairsDist['%d_%d' % (stopFr, stopTo)]
            if dist <= const.maxWalkDist:
                walkTime = int(dist / const.walkSpeed + .5)  # in seconds
                shortestPaths.append([stopFr, stopTo, timeWindow[0], timeWindow[0] + walkTime, [], -1])
                continue

            #foundPath = False
            #starttimedfStopFrNds = time.perf_counter()
            #nNdFrTried = 0
            for row in nodesFrNP:
                #nNdFrTried += 1
                nodeFr = row[0]
                timeFr = row[1]
                dfStopToNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopTo) &
                                              (dfNdDetails[dfNdDetails_time] >= timeFr + const.maxWalkTime) &
                                              (dfNdDetails[dfNdDetails_time] <= timeWindow[1]) &
                                              (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_arr)]
                dfStopToNds = dfStopToNds.sort_values(by = [dfNdDetails_time])
                nodeToList = dfStopToNds[dfNdDetails_ndID].values.tolist()

                # calc shortest paths (in terms of number of transfers) from nodeFr to nodeToList
                # Note: the returned paths in pathsFound are in the order of nodeToList,
                # i.e. in chronological order of nodeTo
                pathsFound = igraph.GraphBase.get_shortest_paths(gttTransit, v=nodeFr, to=nodeToList,
                                                                    weights=const.GttEdgeAttribs.transfer.name)
                # gets number of elements of each path in pathsFound
                pathLen = [len(a) for a in pathsFound]
                # gets 1st value in pathLen larger than 1, returns -1 if no value in pathLen larger than 1
                # e.g. pathLen = [1,1,4,2,1,5,4] will return 4
                # e.g. pathLen = [1,1,1,1,1,1] will return -1
                firstValidPathLen = next((x for x in pathLen if x>1), -1)
                if firstValidPathLen > -1:
                    #foundPath = True
                    # gets the 1st index of firstValidPathLen in pathLen, which is also the index in pathsFound
                    # e.g. if pathLen = [1,1,4,2,1,5,4], pathLen.index(4) will return 2
                    iPath = pathLen.index(firstValidPathLen)
                    path = pathsFound[iPath]
                    # gets stopTo and timeTo from the last node in the selected path
                    nodeTo = path[-1]
                    timeTo = int(gttTransit.vs[nodeTo]['ndDesc'].split('_')[3])
                    shortestPaths.append([stopFr, stopTo, timeFr, timeTo, path, -2])
                    break

            #if foundPath==True:
            #    print('\t\tpathFound between stopFr %d - stopTo %d, %d/%d nodeFrs tried, took %.4g secs' %
            #          (stopFr, stopTo, nNdFrTried, nodesFrNP[:,0].size, time.perf_counter()-starttimedfStopFrNds))
            #else:
            #    print('\t\tpathNotFound between stopFr %d - stopTo %d, %d/%d nodeFrs tried, took %.4g secs' %
            #          (stopFr, stopTo, nNdFrTried, nodesFrNP[:,0].size, time.perf_counter()-starttimedfStopFrNds))

            countStopTo += 1
            if countStopTo % 2000 == 0:
                duration = time.perf_counter() - starttime
                print('\tprocess %d- %d of %d stopTo finished - elapsed time %.4g secs' %
                      (procNumber, countStopTo, len(allStops), duration))

        countStopFr += 1
        duration = time.perf_counter() - starttime
        print('process %d - %d stopFr completed (of %d stopFr to %d stopTo) - elapsed time %.4g mins (%.4g secs)' %
              (procNumber, countStopFr, len(stopFrList), len(allStops), duration / 60, duration))

    with open('%s/shortestPaths_%d.pkl' % (gttShortestPathsFolder, procNumber), 'wb') as f:
        pickle.dump(shortestPaths, f)

# ======================================================================================================================
def calcShortestPathsSubset_v2(gttTransit, stopFrList, allStops, timeWindow, dfNdDetails,
                            stopPairsDist, procNumber, gttShortestPathsFolder):
    # defines dfNdDetails columns
    dfNdDetails_stnID = const.dfNdDetailsCols.StationId.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name
    dfNdDetails_arrDep = const.dfNdDetailsCols.arrOrDep.name
    dfNdDetails_arr = const.dfNdDetailsCols.arr.name
    dfNdDetails_dep = const.dfNdDetailsCols.dep.name
    dfNdDetails_ndID = const.dfNdDetailsCols.gttNdID.name

    shortestPaths = []
    countStopFr = 0
    starttime = time.perf_counter()
    for stopFr in stopFrList:
        countStopTo = 0
        for stopTo in allStops:
            if stopFr == stopTo: continue

            dist = stopPairsDist['%d_%d' % (stopFr, stopTo)]
            if dist <= const.maxWalkDist:
                walkTime = int(dist / const.walkSpeed + .5)  # in seconds
                shortestPaths.append([stopFr, stopTo, timeWindow[0], timeWindow[0] + walkTime, [], -1])
                continue

            dfStopFrNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopFr) &
                                          (dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                          (dfNdDetails[dfNdDetails_time] <= timeWindow[1]) &
                                          (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_dep)]
            dfStopFrNds = dfStopFrNds.sort_values(by = [dfNdDetails_time])

            nNdFrTried = 0
            foundPath = False
            starttimedfStopFrNds = time.perf_counter()
            for idxFr, rowFr in dfStopFrNds.iterrows():
                nNdFrTried += 1
                nodeFr= rowFr[dfNdDetails_ndID]
                stopFr = rowFr[dfNdDetails_stnID]
                timeFr = rowFr[dfNdDetails_time]
                dfStopToNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopTo) &
                                              (dfNdDetails[dfNdDetails_time] >= timeFr + const.maxWalkTime) &
                                              (dfNdDetails[dfNdDetails_time] <= timeWindow[1]) &
                                              (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_arr)]
                dfStopToNds = dfStopToNds.sort_values(by = [dfNdDetails_time])
                nodeToList = dfStopToNds[dfNdDetails_ndID].values.tolist()

                # calc shortest paths (in terms of number of transfers) from nodeFr to nodeToList
                # Note: the returned paths in pathsFound are in the order of nodeToList,
                # i.e. in chronological order of nodeTo
                pathsFound = igraph.GraphBase.get_shortest_paths(gttTransit, v=nodeFr, to=nodeToList,
                                                                    weights=const.GttEdgeAttribs.transfer.name)
                # gets number of elements of each path in pathsFound
                pathLen = [len(a) for a in pathsFound]
                # gets 1st value in pathLen larger than 1, returns -1 if no value in pathLen larger than 1
                # e.g. pathLen = [1,1,4,2,1,5,4] will return 4
                # e.g. pathLen = [1,1,1,1,1,1] will return -1
                firstValidPathLen = next((x for x in pathLen if x>1), -1)
                if firstValidPathLen > -1:
                    foundPath = True
                    # gets the 1st index of firstValidPathLen in pathLen, which is also the index in pathsFound
                    # e.g. if pathLen = [1,1,4,2,1,5,4], pathLen.index(4) will return 2
                    iPath = pathLen.index(firstValidPathLen)
                    path = pathsFound[iPath]
                    # gets stopTo and timeTo from the last node in the selected path
                    nodeTo = path[-1]
                    timeTo = int(gttTransit.vs[nodeTo]['ndDesc'].split('_')[3])
                    shortestPaths.append([stopFr, stopTo, timeFr, timeTo, path, -2])
                    break

            if foundPath==True:
                print('\t\tpathFound between stopFr %d - stopTo %d, %d/%d nodeFrs tried, took %.4g secs' %
                      (stopFr, stopTo, nNdFrTried, dfStopFrNds.shape[0], time.perf_counter()-starttimedfStopFrNds))
            else:
                print('\t\tpathNotFound between stopFr %d - stopTo %d, %d/%d nodeFrs tried, took %.4g secs' %
                      (stopFr, stopTo, nNdFrTried, dfStopFrNds.shape[0], time.perf_counter()-starttimedfStopFrNds))

            countStopTo += 1
            if countStopTo % 10 == 0:
                duration = time.perf_counter() - starttime
                print('\t%d of %d stopTo finished - elapsed time %.4g secs' % (countStopTo, len(allStops), duration))

        countStopFr += 1
        duration = time.perf_counter() - starttime
        print('process %d - %d stopFr completed (of %d stopFr to %d stopTo) - elapsed time %.4g mins (%.4g secs)' %
              (procNumber, countStopFr, len(stopFrList), len(allStops), duration / 60, duration))

    with open('%s/shortestPaths_%d.pkl' % (gttShortestPathsFolder, procNumber), 'wb') as f:
        pickle.dump(shortestPaths, f)

# ======================================================================================================================
def calcShortestPathsSubset(gttTransit, stopFrList, allStops, timeWindow, dfNdDetails,
                            stopPairsDist, procNumber, gttShortestPathsFolder):
    # defines dfNdDetails columns
    dfNdDetails_stnID = const.dfNdDetailsCols.StationId.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name
    dfNdDetails_arrDep = const.dfNdDetailsCols.arrOrDep.name
    dfNdDetails_arr = const.dfNdDetailsCols.arr.name
    dfNdDetails_dep = const.dfNdDetailsCols.dep.name
    dfNdDetails_ndID = const.dfNdDetailsCols.gttNdID.name

    shortestPaths = []
    countStopFr = 0
    starttime = time.perf_counter()
    for stopFr in stopFrList:
        for stopTo in allStops:
            if stopFr==stopTo: continue

            dist = stopPairsDist['%d_%d' % (stopFr, stopTo)]
            if dist <= const.maxWalkDist:
                walkTime = int(dist / const.walkSpeed + .5)  # in seconds
                shortestPaths.append([stopFr, stopTo, timeWindow[0], timeWindow[0] + walkTime, '', -1])
                continue

            dfStopToNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopTo) &
                                          (dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                          (dfNdDetails[dfNdDetails_time] <= timeWindow[1]) &
                                          (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_arr)]
            dfStopToNds = dfStopToNds.sort_values(by = [dfNdDetails_time])

            for idxTo, rowTo in dfStopToNds.iterrows():
                nodeTo = rowTo[dfNdDetails_ndID]
                timeTo = rowTo[dfNdDetails_time]
                # gets the list of departure nodes at stopFr sorted in chronological order
                dfStopFrNds = dfNdDetails.loc[(dfNdDetails[dfNdDetails_stnID] == stopFr) &
                                              (dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                              (dfNdDetails[dfNdDetails_time] <= timeTo - const.maxWalkTime) &
                                              (dfNdDetails[dfNdDetails_arrDep] == dfNdDetails_dep)]
                dfStopFrNds = dfStopFrNds.sort_values(by = [dfNdDetails_time])

                pathFound = False
                for idxFr, rowFr in dfStopFrNds.iterrows():
                    nodeFr = rowFr[dfNdDetails_ndID]
                    timeFr = rowFr[dfNdDetails_time]
                    path = igraph.GraphBase.get_shortest_paths(gttTransit, nodeFr, nodeTo,
                                                               weights=const.GttEdgeAttribs.transfer.name)
                    if len(path[0])>1:
                        shortestPaths.append([stopFr, stopTo, timeFr, timeTo, convert2Str(path[0]), -2])
                        pathFound = True
                        break
                if pathFound == True:
                    break

        countStopFr += 1
        if countStopFr % 10 == 0:
            duration = time.perf_counter() - starttime
            print('process %d - %d stopFr completed (out of %d stopFr to %d stopTo) - took %.4g mins (%.4g secs)' %
                  (procNumber, countStopFr, len(stopFrList), len(allStops), duration/60, duration))

    with open('%s/shortestPaths_%d.pkl' % (gttShortestPathsFolder, procNumber), 'wb') as f:
        pickle.dump(shortestPaths, f)


# ======================================================================================================================
def calcShortestPathsMultip(gttTransit, dfNdDetails, dfAllTransfers, timeWindow, nProcesses, stopPairsDist,
                            gttShortestPathsFolder):
    # defines dfNdDetails columns
    dfNdDetails_stnID = const.dfNdDetailsCols.StationId.name
    dfNdDetails_time = const.dfNdDetailsCols.time.name

    # adds transfer edges to gttTransit
    starttime = time.perf_counter()
    addTransferEdgesAllStops(gttTransit, dfAllTransfers, timeWindow)
    print('addTransferEdgesAllStops finished - took %.4g seconds' % (time.perf_counter()-starttime))

    # gets the list of stops within timeWindow
    allStops = dfNdDetails[dfNdDetails_stnID].loc[(dfNdDetails[dfNdDetails_time] >= timeWindow[0]) &
                                                  (dfNdDetails[dfNdDetails_time] <= timeWindow[1])].values.tolist()
    allStops = list(set(allStops))  # removes duplicates in stationsSel

    # splits allStops into nProcesses paths
    stopParts = partition(allStops, nProcesses)
    processes = []
    for i in range(nProcesses):
        stopFrList = stopParts[i]
        p = multiprocessing.Process(target = calcShortestPathsSubset_v3,
                                    args=(gttTransit, stopFrList, allStops, timeWindow, dfNdDetails,
                                          stopPairsDist, i, gttShortestPathsFolder))
        processes.append(p)
        p.start()
        print('calcShortestPathsMultip process %d started - %d stopFr' % (i, len(stopFrList)))

    for p in processes:
        p.join()

# ======================================================================================================================
def main_ttgAnalysisMultip_v2():
    gttTransit = pickle.load(open('%s/GttTransit_v2.pkl' % const.picklesFolder, 'rb'))
    dfNdDetails = pickle.load(open('%s/dfNdDetails_v2.pkl' % const.picklesFolder, 'rb'))
    dfAllTransfers = pickle.load(open('%s/dfAllTransfers.pkl', 'rb'))
    stopPairsDist = pickle.load(open('%s/stopPairsDist.pkl' % const.picklesFolder, 'rb'))

    startHr = 5
    endHr = 8
    timeWindow = [startHr * 3600, endHr * 3600]
    nProcesses = 56

    startHrStr = '0%d00' % startHr if startHr < 10 else '%d00' % startHr
    endHrStr = '0%d00' % endHr if endHr < 10 else '%d00' % endHr
    gttShortestPathsFolder = '%s/gttShortestPaths/%s_%s' % (const.trashSite, startHrStr, endHrStr)
    if os.path.exists(gttShortestPathsFolder) == False:
        os.mkdir(gttShortestPathsFolder)

    calcShortestPathsMultip(gttTransit, dfNdDetails, dfAllTransfers, timeWindow, nProcesses, stopPairsDist,
                            gttShortestPathsFolder)

