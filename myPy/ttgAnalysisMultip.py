import networkx as nx
import pandas as pd
import random
import multiprocessing

import constants as const
import utilities as utils

# ======================================================================================================================
def getGttSub_v2(Gtt, ndIDict, timeWindow):
    selectedEdges = []
    GttSubStops = []

    for edge in Gtt.edges:
        timeFr = int(ndIDict[edge[0]].split('_')[3])
        stopFr = int(ndIDict[edge[0]].split('_')[2])
        stopTo = int(ndIDict[edge[1]].split('_')[2])
        if timeFr >= timeWindow[0] and timeFr <= timeWindow[1]:
            selectedEdges.append(edge)
            if stopFr not in GttSubStops: GttSubStops.append(stopFr)
            if stopTo not in GttSubStops: GttSubStops.append(stopTo)
    # creates a subgraph from selectedEdges
    GttSub = Gtt.edge_subgraph(selectedEdges).copy()

    return GttSub, GttSubStops

# ======================================================================================================================
def mkWaitEdgesList(GttSubStops, dfNdDetails, timeWindow, filename):
    countStops = 0
    waitEdges = pd.DataFrame()
    for stop in GttSubStops:
        dfStopArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stop) &
                                      (dfNdDetails[const.dfNdDetailsCols.time.name] >= timeWindow[0]) &
                                      (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                      (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                       const.dfNdDetailsCols.arr.name)]
        uDirRoutes = dfStopArrNd[const.dfNdDetailsCols.dirRoute.name].unique()
        if uDirRoutes.shape[0] == 1:  # there is only one directed route passing this node
            continue

        for idxFr,rowFr in dfStopArrNd.iterrows():
            nodeFr = rowFr[const.dfNdDetailsCols.gttNdID.name]
            dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
            timeFr = rowFr[const.dfNdDetailsCols.time.name]

            dfStopDepNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stop) &
                                          (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr) &
                                          (dfNdDetails[const.dfNdDetailsCols.time.name] <= min(timeFr + const.maxWaitTime, timeWindow[1])) &
                                          (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                          (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                           const.dfNdDetailsCols.dep.name)]
            dfStopDepNd = dfStopDepNd.assign(nodeFr=nodeFr)
            waitEdges = pd.concat([waitEdges, dfStopDepNd[['nodeFr', const.dfNdDetailsCols.gttNdID.name]]])

        countStops += 1
        if countStops%100==0:
            print('addWaitEdges for %d stops completed (out of %d stops)' % (countStops, len(GttSubStops)))

    waitEdges.to_csv(filename, index=False)

# ======================================================================================================================
def mkWalkEdgesList(GttSubStops, dfNdDetails, G, timeWindow, filename):
    countStopFr = 0
    walkEdges = pd.DataFrame()
    for stopFr in GttSubStops:
        dfStopFrArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
                                        (dfNdDetails[const.dfNdDetailsCols.time.name] >= timeWindow[0]) &
                                        (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                        (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                         const.dfNdDetailsCols.arr.name)]
        stopFrCoord = [G.nodes[stopFr][const.GNodeAttribs.Lng.name], G.nodes[stopFr][const.GNodeAttribs.Lat.name]]
        for stopTo in GttSubStops:
            if stopFr == stopTo: continue
            stopToCoord = [G.nodes[stopTo][const.GNodeAttribs.Lng.name], G.nodes[stopTo][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(stopFrCoord, stopToCoord)  # distance in metres
            # if stopFr and stopTo are far enough from each other, don't build walk link between them
            # moves on to the next stopTo
            if dist > const.maxWalkDist: continue

            # calculates walk time based on the distance calculated above
            walkTime = int(dist / const.walkSpeed + .5)  # in seconds

            for idxFr, rowFr in dfStopFrArrNd.iterrows():
                nodeFr = rowFr[const.dfNdDetailsCols.gttNdID.name]
                dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
                timeFr = rowFr[const.dfNdDetailsCols.time.name]
                dfStopToDepNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopTo) &
                                                (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr + walkTime) &
                                                (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                                (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                                (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                 const.dfNdDetailsCols.dep.name)]

                dfStopToDepNd = dfStopToDepNd.assign(nodeFr=nodeFr)
                walkEdges = pd.concat([walkEdges, dfStopToDepNd[['nodeFr', const.dfNdDetailsCols.gttNdID.name]]])

        countStopFr += 1
        if countStopFr%100==0:
            print('addWalkEdges for %d stopFr completed' % countStopFr)

    walkEdges.to_csv(filename, index=False)

# ======================================================================================================================
def calcShortestPathsSubGraph(GttSub, GttSubStops, dfNdDetails, timeWindow, G, ndPairsTried):
    shortTimesGttSub = []
    # now that we have GttSub with transit edges, walk edges and wait egdes, we can start find shortest paths
    countStopFr = 0
    for stopFr in GttSubStops:
        stopFrCoord = [G.nodes[stopFr][const.GNodeAttribs.Lng.name], G.nodes[stopFr][const.GNodeAttribs.Lat.name]]
        # gets the list of departure nodes at stopFr sorted in chronological order
        dfStopFrDepNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
                                        (dfNdDetails[const.dfNdDetailsCols.time.name] >= timeWindow[0]) &
                                        (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                        (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                         const.dfNdDetailsCols.dep.name)]
        dfStopFrDepNd = dfStopFrDepNd.sort_values(by=[const.dfNdDetailsCols.time.name])

        for stopTo in GttSubStops:
            if stopTo == stopFr: continue

            stopToCoord = [G.nodes[stopTo][const.GNodeAttribs.Lng.name], G.nodes[stopTo][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(stopFrCoord, stopToCoord)  # distance in metres
            # if stopFr and stopTo are far enough from each other, don't build walk link between them
            # moves on to the next stopTo
            if dist <= const.maxWalkDist:
                # calculates walk time based on the distance calculated above
                walkTime = int(dist / const.walkSpeed + .5)  # in seconds
                shortTimesGttSub.append([stopFr, stopTo, timeWindow[0], timeWindow[0] + walkTime,
                                         '_%d_%d' % (stopFr, stopTo), -1])
                continue

            # stopTo is beyond walk distance from stopFr.
            # finds shortest path to stopTo for each selected departure node at stopFr
            for idxFr, rowFr in dfStopFrDepNd.iterrows():
                nodeFr = rowFr[const.dfNdDetailsCols.gttNdID.name]
                timeFr = rowFr[const.dfNdDetailsCols.time.name]
                # gets the list of arrival nodes at stopTo
                dfStopToArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopTo) &
                                                (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr) &
                                                (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                                (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                 const.dfNdDetailsCols.arr.name)]
                dfStopToArrNd = dfStopToArrNd.sort_values(by=[const.dfNdDetailsCols.time.name])
                for idxTo, rowTo in dfStopToArrNd.iterrows():
                    nodeTo = rowTo[const.dfNdDetailsCols.gttNdID.name]
                    timeTo = rowTo[const.dfNdDetailsCols.time.name]
                    if '%d_%d_%d_%d' % (stopFr, stopTo, timeFr, timeTo) not in ndPairsTried:
                        # finds the shortest path between rowFr and rowTo in terms of number of transfers
                        try:
                            nTransfers, path = nx.single_source_dijkstra(GttSub, nodeFr, nodeTo)
                            shortTimesGttSub.append([stopFr, stopTo, timeFr, timeTo, convert2Str(path), nTransfers])
                            ndPairsTried.append('%d_%d_%d_%d' % (stopFr, stopTo, timeFr, timeTo))
                            break
                        except nx.NetworkXNoPath:
                            ndPairsTried.append('%d_%d_%d_%d' % (stopFr, stopTo, timeFr, timeTo))

        countStopFr += 1
        if countStopFr % 10 == 0:
            print('%d stopFr completed (out of %d)' % (countStopFr, len(GttSubStops)))

    return GttSub, shortTimesGttSub, ndPairsTried

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
def mkWaitEdgesMultip(Gtt, dfNdDetails, ndIDict, timeWindow, nProcesses):
    # extracts all edges that has DEPARTURE time within timeWindow
    # also gets the list of all stops in GttSub
    GttSub, GttSubStops = getGttSub_v2(Gtt, ndIDict, timeWindow)
    print('finsihed getGttSub_v2')

    # randomly split GttSubStops into nProcesses parts
    GttSubStopsParts = partition(GttSubStops, nProcesses)

    # creates parallel processes to create wait edges
    waitEdgeProcesses = []
    for i in range(nProcesses):
        GttSubStopsPart = GttSubStopsParts[i]
        p = multiprocessing.Process(target=mkWaitEdgesList,
                                    args=(GttSubStopsPart, dfNdDetails, timeWindow,
                                          '%s/waitEdges_%d.csv' % (const.trashSite, i)))
        waitEdgeProcesses.append(p)
        p.start()

    for p in waitEdgeProcesses:
        p.join()

    print('finsihed mkWaitEdgesList')


# ======================================================================================================================
def mkWalkEdgesMultip(Gtt, dfNdDetails, ndIDict, timeWindow, G, nProcesses):
    # extracts all edges that has DEPARTURE time within timeWindow
    # also gets the list of all stops in GttSub
    GttSub, GttSubStops = getGttSub_v2(Gtt, ndIDict, timeWindow)
    print('finsihed getGttSub_v2')

    # randomly split GttSubStops into nProcesses parts
    GttSubStopsParts = partition(GttSubStops, nProcesses)

    # creates parallel processes to create walk edges
    walkEdgeProcesses = []
    for i in range(nProcesses):
        GttSubStopsPart = GttSubStopsParts[i]
        p = multiprocessing.Process(target=mkWalkEdgesList,
                                    args=(GttSubStopsPart, dfNdDetails, G, timeWindow,
                                          '%s/walkEdges_%d.csv' % (const.trashSite, i)))
        walkEdgeProcesses.append(p)
        p.start()
    for p in walkEdgeProcesses:
        p.join()
    print('finsihed mkWalkEdgesList')

# ======================================================================================================================
def addWalkAndWaitEdges(GttSub, waitFiles, walkFiles, G, ndIDict):
    for file in waitFiles:
        dfWaitEdges = pd.read_csv(file)
        for id, row in dfWaitEdges.iterrows():
            nodeFr = row['nodeFr']
            timeFr = int(ndIDict[nodeFr].split('_')[3])
            nodeTo = row[const.dfNdDetailsCols.gttNdID.name]
            timeTo = int(ndIDict[nodeTo].split('_')[3])
            GttSub.add_edges(nodeFr, nodeTo)
            GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWait.value
            GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.time.name] = timeTo - timeFr
            GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.transfer.name] = 1
            GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.dirRouteID.name] = ''


# ======================================================================================================================
def calcShortestPaths(Gtt, dfNdDetails, ndIDict, timeWindow, G, ndPairsTried):
    '''
    :param Gtt: GtopoTemp
    :param dfNdDetails: [RouteId, StationDirection, StationId, 'time', 'dirRoute', 'arrOrDep'] %d, %d, %d, %d, %s, %s
    :param timeWindow: [startTimeInSecs, endTimeInSecs]
    :return:
    '''
    nProcesses = 8
    mkWaitEdgesMultip(Gtt, dfNdDetails, ndIDict, timeWindow, nProcesses)

    mkWalkEdgesMultip(Gtt, dfNdDetails, ndIDict, timeWindow, G, nProcesses)

    # adds wait edges and walk edges to the Gtt
    #addWalkWaitEdges(GttSub, walkFiles, waitFiles)

    # calculates shortest paths
    '''
    GttSub, shortestTimes, ndPairsTried = calcShortestPathsSubGraph(GttSub, GttSubStops, dfNdDetails,
                                                                       timeWindow, G, ndPairsTried)
    print('finished calcShortestPathsSubGraph')
    return Gtt, shortestTimes, ndPairsTried
    '''