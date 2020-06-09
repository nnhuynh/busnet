import networkx as nx

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
def getGttSub(Gtt, timeWindow):
    selectedEdges = []
    GttSubStops = []
    for edge in Gtt.edges:
        timeFr = int(edge[0].split('_')[3])
        stopFr = int(edge[0].split('_')[2])
        stopTo = int(edge[1].split('_')[2])
        if timeFr >= timeWindow[0] and timeFr <= timeWindow[1]:
            selectedEdges.append(edge)
            if stopFr not in GttSubStops: GttSubStops.append(stopFr)
            if stopTo not in GttSubStops: GttSubStops.append(stopTo)
    # creates a subgraph from selectedEdges
    GttSub = Gtt.edge_subgraph(selectedEdges)

    return GttSub, GttSubStops

# ======================================================================================================================
def addWaitEdges_v2(GttSub, GttSubStops, dfNdDetails):
    countStops = 0
    for stop in GttSubStops:
        dfStopArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stop) &
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
                                          (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeFr + const.maxWaitTime) &
                                          (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                          (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                           const.dfNdDetailsCols.dep.name)]
            for idxTo, rowTo in dfStopDepNd.iterrows():
                nodeTo = rowTo[const.dfNdDetailsCols.gttNdID.name]
                timeTo = rowTo[const.dfNdDetailsCols.time.name]
                GttSub.add_edge(nodeFr, nodeTo)
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWait.value
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.time.name] = timeTo - timeFr
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.transfer.name] = 1
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.dirRouteID.name] = ''

        countStops += 1
        if countStops%100==0:
            print('addWaitEdges for %d stops completed (out of %d stops)' % (countStops, len(GttSubStops)))

# ======================================================================================================================
def addWalkEdges_v2(GttSub, GttSubStops, dfNdDetails, G, timeWindow):
    countStopFr = 0
    for stopFr in GttSubStops:
        dfStopFrArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
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
                                                (dfNdDetails[const.dfNdDetailsCols.dirRoute.name != dirRouteFr]) &
                                                (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                 const.dfNdDetailsCols.dep.name)]
                for idxTo, rowTo in dfStopToDepNd.iterrows():
                    nodeTo = rowTo[const.dfNdDetailsCols.gttNdID.name]
                    timeTo = rowTo[const.dfNdDetailsCols.time.name]
                    GttSub.add_edge(nodeFr, nodeTo)
                    GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWalk.value
                    GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.time.name] = timeTo - timeFr
                    GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.transfer.name] = 1
                    GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.dirRouteID.name] = ''

                '''
                # THE BELOW FEATURE WILL BE ADDED IN THE FUTURE - NOT IN THE FIRST PAPER
                # adds a node at stopTo which represents the end point of a walk link from nodeFr and which doesn't
                # belong to any dirRoute. This represents the case where a person arrives by bus at stopFr and walks to
                # their final destination at stopTo.
                # However, we don't add this node if 
                # - there is an arrival node at stopTo belonging to the same dirRoute of nodeFr at a time less than 5 
                # minutes from timeFr+walkTime
                '''
    countStopFr += 1
    if countStopFr%10==0:
        print('addWalkEdges for %d stopFr completed' % countStopFr)


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
def calcShortestPaths(Gtt, dfNdDetails, ndIDict, timeWindow, G, ndPairsTried):
    '''
    :param Gtt: GtopoTemp
    :param dfNdDetails: [RouteId, StationDirection, StationId, 'time', 'dirRoute', 'arrOrDep'] %d, %d, %d, %d, %s, %s
    :param timeWindow: [startTimeInSecs, endTimeInSecs]
    :return:
    '''
    # extracts all edges that has DEPARTURE time within timeWindow
    # also gets the list of all stops in GttSub
    GttSub, GttSubStops = getGttSub_v2(Gtt, ndIDict, timeWindow)
    print('finsihed getGttSub_v2')

    # adds wait links for each stop in GttSub
    addWaitEdges_v2(GttSub, GttSubStops, dfNdDetails)
    print('finsihed addWaitEdges_v2')

    # adds walk links in GttSub
    addWalkEdges_v2(GttSub, GttSubStops, dfNdDetails, G, timeWindow)
    print('finsihed addWalkEdges_v2')

    GttSub, shortestTimes, ndPairsTried = calcShortestPathsSubGraph(GttSub, GttSubStops, dfNdDetails,
                                                                       timeWindow, G, ndPairsTried)
    print('finished calcShortestPathsSubGraph')
    return Gtt, shortestTimes, ndPairsTried


# THE BELOW FUNCTIONS ARE DEPRECATED ===================================================================================
# ======================================================================================================================
def addWaitEdges(GttSub, GttSubStops, dfNdDetails):
    for stop in GttSubStops:
        dfStopArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name]==stop) &
                                      (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name]==
                                       const.dfNdDetailsCols.arr.name)]
        uDirRoutes = dfStopArrNd[const.dfNdDetailsCols.dirRoute.name].unique()
        if uDirRoutes.shape[0] == 1:  # there is only one directed route passing this node
            continue

        for idxFr,rowFr in dfStopArrNd.iterrows():
            nodeFr = '%s_%s_%s_%s' % (rowFr[const.dfNdDetailsCols.RouteId.name],
                                      rowFr[const.dfNdDetailsCols.StationDirection.name],
                                      rowFr[const.dfNdDetailsCols.StationId.name],
                                      rowFr[const.dfNdDetailsCols.time.name])

            dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
            timeFr = rowFr[const.dfNdDetailsCols.time.name]
            dfStopDepNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name]==stop) &
                                          (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr) &
                                          (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                          (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name]==
                                           const.dfNdDetailsCols.dep.name)]
            for idxTo, rowTo in dfStopDepNd.iterrows():
                nodeTo = '%s_%s_%s_%s' % (rowTo[const.dfNdDetailsCols.RouteId.name],
                                          rowTo[const.dfNdDetailsCols.StationDirection.name],
                                          rowTo[const.dfNdDetailsCols.StationId.name],
                                          rowTo[const.dfNdDetailsCols.time.name])
                GttSub.add_edge(nodeFr, nodeTo)
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWait.value

# ======================================================================================================================
def addWalkEdges(GttSub, GttSubStops, dfNdDetails, G):
    for stopFr in GttSubStops:
        dfStopFrArrNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
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
                nodeFr = '%s_%s_%s_%s' % (rowFr[const.dfNdDetailsCols.RouteId.name],
                                          rowFr[const.dfNdDetailsCols.StationDirection.name],
                                          rowFr[const.dfNdDetailsCols.StationId.name],
                                          rowFr[const.dfNdDetailsCols.time.name])

                dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
                timeFr = rowFr[const.dfNdDetailsCols.time.name]
                dfStopToDepNd = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopTo) &
                                                (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr + walkTime) &
                                                (dfNdDetails[const.dfNdDetailsCols.dirRoute.name != dirRouteFr]) &
                                                (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                 const.dfNdDetailsCols.dep.name)]
                for idxTo, rowTo in dfStopToDepNd.iterrows():
                    nodeTo = '%s_%s_%s_%s' % (rowTo[const.dfNdDetailsCols.RouteId.name],
                                              rowTo[const.dfNdDetailsCols.StationDirection.name],
                                              rowTo[const.dfNdDetailsCols.StationId.name],
                                              rowTo[const.dfNdDetailsCols.time.name])
                    GttSub.add_edge(nodeFr, nodeTo)
                    GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWalk.value

                # adds a node at stopTo which represents a walk from nodeFr and which doesn't belong to any dirRoute.
                # this represents the case where a person arrives by bus at stopFr and walks to their final destination
                # at stopTo.
                nodeTo = '%s_%s_%s_%s' % (0, 0, rowTo[const.dfNdDetailsCols.StationId.name], timeFr + walkTime)
                GttSub.add_edge(nodeFr, nodeTo)
                GttSub.edges[nodeFr, nodeTo][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeWalk.value
