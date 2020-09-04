import pandas as pd
import re
import networkx as nx
import json
import os

import utilities as utils
import constants as const

# ======================================================================================================================
def construct_G_dfRoutes():
    # initialises graph nodes, graph edges, and dfRoutes
    # assigns all node attributes
    # assigns attributes 'RouteId', 'RouteVarId', 'edgeOrder', 'pathPoints', 'roadDist' to G.edges[edge]['routes']
    # assigns attributes 'RouteId', 'RouteVarId', 'tripDist', 'designedTtb' and 'actualTtb' to dfRoutes
    jsFilesStopSeq = ['../data/stopSeq/%s' % f for f in os.listdir('../data/stopSeq/')]
    G, dfRoutes = makeNetwork(jsFilesStopSeq)

    # assigns values to routeNo, routeName, tripTime in dfRoutes
    updateRouteInfo(dfRoutes, '../data/allRouteInfo.json')

    # assigns 'timetable' to dfRoutes['designedTtb']
    jsFilesRouteInfo = ['../data/routeInfo/%s' % f for f in os.listdir('../data/routeInfo/')]
    updateRouteTimetable(dfRoutes, jsFilesRouteInfo, G)
    # print(dfRoutes[['RouteId', 'RouteVarId', 'RouteNo', 'RouteName', 'tripTime', 'tripDist']])

    # assigns attributes 'nLines', 'meanRoadDist', 'meanTravTime', 'nServices' to edges
    calcMoreEdgeAttribs(G, dfRoutes)

    # removes stops that are outside of HCMC, defined in const.stopsOutsideHCMC
    removeStopsOutsideHCMC(G, dfRoutes)

    # assigns attributes nLines and nServices passing each bus to nodes
    calcNodeStrength(G, dfRoutes)
    calcNodeDegree(G)

    # determines the list of nodes within walk distance to each node
    #calcNearbyNodes(G)

    return G, dfRoutes

# ======================================================================================================================
def removeStopsOutsideHCMC(G, dfRoutes):
    # removes stops outside of HCMC which are at the BEGINNING of a route, done to both designedTtb and actualTtb.
    # notes that actualTtb has only StationCode and StationOrder, which are identical to those in designedTtb
    for idx, route in dfRoutes.iterrows():
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        if stnList[0] in const.stopsOutsideHCMC: # first stop of this route is outside of HCMC
            '''
            print('1st node out - route %d_%d (%s), nStations %d' % (route[const.RoutesCols.RouteId.name],
                                                                     route[const.RoutesCols.StationDirection.name],
                                                                     route[const.RoutesCols.RouteNo.name],
                                                                     len(stnList)))
            '''
            # removes link from this node from G
            if G.has_edge(stnList[0], stnList[1]):
                G.remove_edge(stnList[0], stnList[1])

            timetable = designedTtb[const.RoutesTtbCols.timetable.name]
            stnOrder = designedTtb[const.RoutesTtbCols.StationOrder.name]
            del stnOrder[0] # removes order of the 1st station from stnOrder
            del stnList[0] # removes id of this first stop from the stnList
            # removes time element of this first stop for each service in the timetable
            for iServ in range(len(timetable)):
                del timetable[iServ][0]

            # updates designedTtb
            designedTtb[const.RoutesTtbCols.StationOrder.name] = stnOrder
            designedTtb[const.RoutesTtbCols.StationCode.name] = stnList
            designedTtb[const.RoutesTtbCols.timetable.name] = timetable
            route[const.RoutesCols.designedTtb.name] = designedTtb
            # updates actualTtb
            actualTtb = route[const.RoutesCols.actualTtb.name]
            actualTtb[const.RoutesTtbCols.StationOrder.name] = stnOrder
            actualTtb[const.RoutesTtbCols.StationCode.name] = stnList
            route[const.RoutesCols.actualTtb.name] = actualTtb


    # removes stops outside of HCMC which are at the END of a route, done to both designedTtb and actualTtb.
    # notes that actualTtb has only StationCode and StationOrder, which are identical to those in designedTtb
    for idx, route in dfRoutes.iterrows():
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        if stnList[-1] in const.stopsOutsideHCMC:
            # removes link to this node from G
            if G.has_edge(stnList[-2], stnList[-1]):
                G.remove_edge(stnList[-2], stnList[-1])
            '''
            print('last node out - route %d_%d (%s), nStations %d' % (route[const.RoutesCols.RouteId.name],
                                                                      route[const.RoutesCols.StationDirection.name],
                                                                      route[const.RoutesCols.RouteNo.name],
                                                                      len(stnList)))
            '''
            timetable = designedTtb[const.RoutesTtbCols.timetable.name]
            stnOrder = designedTtb[const.RoutesTtbCols.StationOrder.name]
            del stnOrder[-1]  # removes order of the 1st station from stnOrder
            del stnList[-1]  # removes id of this first stop from the stnList
            # removes time element of this first stop for each service in the timetable
            for iServ in range(len(timetable)):
                del timetable[iServ][-1]

            # updates designedTtb
            designedTtb[const.RoutesTtbCols.StationOrder.name] = stnOrder
            designedTtb[const.RoutesTtbCols.StationCode.name] = stnList
            designedTtb[const.RoutesTtbCols.timetable.name] = timetable
            route[const.RoutesCols.designedTtb.name] = designedTtb
            # updates actualTtb
            actualTtb = route[const.RoutesCols.actualTtb.name]
            actualTtb[const.RoutesTtbCols.StationOrder.name] = stnOrder
            actualTtb[const.RoutesTtbCols.StationCode.name] = stnList
            route[const.RoutesCols.actualTtb.name] = actualTtb

    # if nodes that are outside of HCMC has no edge and if they are in G, remove them
    for node in const.stopsOutsideHCMC:
        if G.has_node(node)==False: continue # if this node is not in G, can't remove it
        haslink = False
        for otherNode in G.nodes:
            if otherNode!=node and (G.has_edge(node, otherNode) or G.has_edge(otherNode, node)):
                haslink = True
                break
        if haslink==False:
            print('node %d has no links' % node)
            G.remove_node(node)


# ======================================================================================================================
def calcNearbyNodes(G):
    for inode in G.nodes:
        nodeiCoord = [G.nodes[inode][const.GNodeAttribs.Lng.name], G.nodes[inode][const.GNodeAttribs.Lat.name]]
        neighbours = []
        for jnode in G.nodes:
            if jnode==inode: continue
            nodejCoord = [G.nodes[jnode][const.GNodeAttribs.Lng.name], G.nodes[jnode][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(nodeiCoord, nodejCoord)  # distance in metres
            if dist <= const.maxWalkDist:
                neighbours.append(jnode)
        G.nodes[inode][const.GNodeAttribs.neighbourNodes.name] = neighbours

# ======================================================================================================================
def calcTimetable(tripTime, tripDist, depTimes, dfStopSeq, G, routeId, stationDir):
    '''
    :param tripTime: in minutes
    :param tripDist:
    :param depTimes: a list of departure time from the first stop in string format
    :param dfStopSeq: a dataframe with 1 column StationCode, index is StationOrder, sorted by index
    :param G:
    :param routeId:
    :param stationDir:
    :return:
    '''
    # convert trip time to seconds
    tripTime = tripTime * 60
    # let us assume that a bus dwells 6 seconds on average at each stop
    # the actual moving time equals total trip time minus the total dwell time at all stops, except for the last stop.
    movingTripTime = tripTime - (len(dfStopSeq.index)-1) * const.dfaultDwellTime
    avgSpd = tripDist / movingTripTime

    ttbAllDay = []
    for iDepTime in range(len(depTimes)):
        timetable = []
        depTime = int(depTimes[iDepTime])
        # appends arrival time and departure time at the first stop to timetable
        timetable.append((depTime, depTime + const.dfaultDwellTime))

        for i in dfStopSeq.index:
            if i==len(dfStopSeq.index)-1: # skips the last station
                break
            #gets road distance between this stop and the next stop for this routeId and stationDir
            crnStop = dfStopSeq.at[i,'StationCode']
            nxtStop = dfStopSeq.at[i+1,'StationCode']
            dfRoutesThisEdge = G.edges[crnStop,nxtStop][const.GEdgeAttribs.routes.name]
            idx = dfRoutesThisEdge.loc[(dfRoutesThisEdge[const.GEdgeRouteCols.RouteId.name]==routeId) &
                                       (dfRoutesThisEdge[const.GEdgeRouteCols.StationDirection.name]==stationDir)].index.to_list()
            roadDist = dfRoutesThisEdge.at[idx[0],const.GEdgeRouteCols.roadDist.name]
            # updates tTrav of in G.edges[crnStop, nxtStop]
            dfRoutesThisEdge.at[idx[0], const.GEdgeRouteCols.tTrav.name] = round(roadDist/avgSpd)
            G.edges[crnStop, nxtStop][const.GEdgeAttribs.routes.name] = dfRoutesThisEdge
            # arrival time at the next stop
            nxtArrTime = timetable[i][1] + round(roadDist/avgSpd)
            nxtDepTime = nxtArrTime + const.dfaultDwellTime
            timetable.append((nxtArrTime, nxtDepTime))

        #timetable = [(round(pair[0]), round(pair[1])) for pair in timetable]
        ttbAllDay.append(timetable)

    return ttbAllDay

# ======================================================================================================================
def updateRouteTimetable(dfRoutes, jsFilesRouteInfo, G):
    for jsFile in jsFilesRouteInfo:
        jsFilename = jsFile.split('/')[-1]
        if jsFilename in const.excludedRoutes:
            print('updateRouteTimetable - reading in %s - EXCLUDED' % jsFile)
            continue

        print('updateRouteTimetable - reading in %s ...' % jsFile)
        with open(jsFile, 'r', encoding='utf-8') as f:
            jsRouteInfo = json.load(f) # notes that jsRouteInfo is a dictionary

        # updates timetable for outbound trip ==========================================================================
        # gets index of row in dfRoutes that has RouteId equal to route id in jsRouteInfo and StationDir is an ODD
        # number (outbound)
        idxOut = dfRoutes.loc[(dfRoutes[const.RoutesCols.RouteId.name]==jsRouteInfo[const.RouteInfoCols.routeId.name]) &
                              (dfRoutes[const.RoutesCols.StationDirection.name]%int(2)==int(1))].index.to_list()
        if len(idxOut)!=0:
            # gets the list of departure times from first stop, e.g. ['18000','18600','19200','19800']
            depTimesOut = jsRouteInfo[const.RouteInfoCols.timeTableOut.name].split(',')
            tripTime = dfRoutes.at[idxOut[0],const.RoutesCols.tripTime.name]
            tripDist = dfRoutes.at[idxOut[0], const.RoutesCols.tripDist.name]
            routeId = dfRoutes.at[idxOut[0],const.RoutesCols.RouteId.name]
            stationDir = dfRoutes.at[idxOut[0],const.RoutesCols.StationDirection.name]
            dfStopSeq = pd.DataFrame.from_dict(dfRoutes.at[idxOut[0], const.RoutesCols.designedTtb.name])
            dfStopSeq = dfStopSeq.set_index(const.RoutesTtbCols.StationOrder.name)
            dfStopSeq = dfStopSeq.sort_index()
            ttbAllDay = calcTimetable(tripTime, tripDist, depTimesOut, dfStopSeq, G, routeId, stationDir)
            dfRoutes.at[idxOut[0], const.RoutesCols.designedTtb.name]['timetable'] = ttbAllDay

        # updates timetable for inbound trip ===========================================================================
        # gets index of row in dfRoutes that has RouteId equal to route id in jsRouteInfo and StationDir is an EVEN
        # number (inbound)
        idxIn = dfRoutes.loc[(dfRoutes[const.RoutesCols.RouteId.name]==jsRouteInfo[const.RouteInfoCols.routeId.name]) &
                             (dfRoutes[const.RoutesCols.StationDirection.name] % int(2) == int(0))].index.to_list()
        if len(idxIn)!=0:
            depTimesIn = jsRouteInfo[const.RouteInfoCols.timeTableIn.name].split(',')
            tripTime = dfRoutes.at[idxIn[0],const.RoutesCols.tripTime.name]
            tripDist = dfRoutes.at[idxIn[0], const.RoutesCols.tripDist.name]
            routeId = dfRoutes.at[idxIn[0], const.RoutesCols.RouteId.name]
            stationDir = dfRoutes.at[idxIn[0], const.RoutesCols.StationDirection.name]
            dfStopSeq = pd.DataFrame.from_dict(dfRoutes.at[idxIn[0], const.RoutesCols.designedTtb.name])
            dfStopSeq = dfStopSeq.set_index(const.RoutesTtbCols.StationOrder.name)
            dfStopSeq = dfStopSeq.sort_index()
            ttbAllDay = calcTimetable(tripTime, tripDist, depTimesIn, dfStopSeq, G, routeId, stationDir)
            dfRoutes.at[idxIn[0], const.RoutesCols.designedTtb.name]['timetable'] = ttbAllDay

# ======================================================================================================================
def processTripTime(tripTimeStr):
    if '-' in tripTimeStr:
        tripTimes = tripTimeStr.split('-')
        tripTimeInt = (int(tripTimes[0]) + int(tripTimes[1]))/2
    else:
        tripTimeInt = int(tripTimeStr)

    return tripTimeInt

# ======================================================================================================================
def updateRouteInfo(dfRoutes,jsFileAllRouteInfo):
    with open(jsFileAllRouteInfo, 'r', encoding='utf-8') as f:
        jsAllRouteInfo = json.load(f)

    for i in range(len(jsAllRouteInfo)):
        print('updateRouteInfo - reading in routeID %d from %s ...' % (jsAllRouteInfo[i]['RouteId'], jsFileAllRouteInfo))
        routeId = jsAllRouteInfo[i]['RouteId']
        print(jsAllRouteInfo[i])
        tripTime = jsAllRouteInfo[i]['TimeOfTrip']
        routeNo = jsAllRouteInfo[i]['RouteNo']
        routeName = jsAllRouteInfo[i]['RouteName']

        tmpIdx = dfRoutes.loc[dfRoutes['RouteId']==routeId].index.to_list()
        dfRoutes.at[tmpIdx,const.RoutesCols.tripTime.name] = processTripTime(tripTime)
        dfRoutes.at[tmpIdx,const.RoutesCols.RouteNo.name] = routeNo
        dfRoutes.at[tmpIdx, const.RoutesCols.RouteName.name] = routeName


# ======================================================================================================================
def calcMoreEdgeAttribs(G, dfRoutes):
    for edge in G.edges():
        dfRoutesThisEdge = G.edges[edge][const.GEdgeAttribs.routes.name]
        # adds the number of lines going through this edge along its direction
        G.edges[edge][const.GEdgeAttribs.nLines.name] = len(dfRoutesThisEdge.index)
        # adds the average road distance
        G.edges[edge][const.GEdgeAttribs.meanRoadDist.name] = dfRoutesThisEdge[const.GEdgeRouteCols.roadDist.name].mean()
        # adds the average travel time
        G.edges[edge][const.GEdgeAttribs.meanTravTime.name] = dfRoutesThisEdge[const.GEdgeRouteCols.tTrav.name].mean()
        # adds the number of services through this edge along its direction
        # for each routeId and stationDir in dfRoutesThisEdge of this edge
        nServices = 0
        for index,row in dfRoutesThisEdge.iterrows():
            routeId = row[const.GEdgeRouteCols.RouteId.name]
            stationDir = row[const.GEdgeRouteCols.StationDirection.name]
            # gets timetable in dfRoutes
            idx = dfRoutes.loc[(dfRoutes[const.RoutesCols.RouteId.name]==routeId) &
                                (dfRoutes[const.RoutesCols.StationDirection.name]==stationDir)].index.to_list()
            ttbAllDay = dfRoutes.at[idx[0],const.RoutesCols.designedTtb.name]['timetable']
            nServices += len(ttbAllDay)
        G.edges[edge][const.GEdgeAttribs.nServices.name] = nServices


# ======================================================================================================================
def calcNodeStrength(G, dfRoutes):
    '''
    calculates strength of a node as the sum of
    - the number of lines (or the number of services) passing that node
    - the number of lines (or the number of services) having that node as the 1st stop
    - the number of lines (or the number of services) having that node as the last stop
    :param G:
    :return:
    '''
    for node in G.nodes:
        dfRoutesThisNode = G.nodes[node][const.GNodeAttribs.routes.name]
        # number of lines passing through this node equals the number rows in dfRoutesThisNode
        G.nodes[node][const.GNodeAttribs.nLines.name] = dfRoutesThisNode.shape[0]
        # number of services
        nServicesThisNode = 0
        for index,row in dfRoutesThisNode.iterrows():
            routeID = row[const.GNodeRouteCols.RouteId.name]
            stnDir = row[const.GNodeRouteCols.StationDirection.name]
            designedTtbDict = dfRoutes[const.RoutesCols.designedTtb.name].\
                loc[(dfRoutes[const.RoutesCols.RouteId.name]==routeID) &
                    (dfRoutes[const.RoutesCols.StationDirection.name]==stnDir)].values[0]
            timetable = designedTtbDict[const.RoutesTtbCols.timetable.name]
            # len(timetable) is the number of services of routeID in direction stnDir
            nServicesThisNode += len(timetable)
        G.nodes[node][const.GNodeAttribs.nServices.name] = nServicesThisNode

# ======================================================================================================================
def calcNodeDegree(G):
    '''
    :param G:
    :return:
    '''
    nodeList = list(G.nodes)
    nodeList.sort()

    # creates adjacency matrix - scipy
    adjMatrix = nx.adjacency_matrix(G, nodelist = nodeList)

    for iNode in range(len(nodeList)):
        node = nodeList[iNode]
        nodeOutDeg = sum([adjMatrix[iNode,i] for i in range(len(nodeList))])
        nodeInDeg = sum([adjMatrix[i,iNode] for i in range(len(nodeList))])
        G.nodes[node][const.GNodeAttribs.inDeg.name] = nodeInDeg
        G.nodes[node][const.GNodeAttribs.outDeg.name] = nodeInDeg
        G.nodes[node][const.GNodeAttribs.totDeg.name] = nodeInDeg + nodeOutDeg

    # creates adjacency matrix - numpy, which will create a matrix of size len(nodeList) x len(nodeList)
    # adjMatrix = nx.to_numpy_matrix(G, nodelist=nodeList)
    # np.savetxt('adjMatrix_np.csv', adjMatrix, delimiter=',')
    # print(adjMatrix)

# ======================================================================================================================
def addNodesEdgesToGraph(G, dfRoutes, dfRawRoute):
    '''
    :param G:
    :param dfRoutes:
    :param dfRawRoute:
    :return:
    '''
    colStationId = const.BusMapRouteCols.StationId.name
    colStationName = const.BusMapRouteCols.StationName.name
    colAddress = const.BusMapRouteCols.Address.name
    colLat = const.BusMapRouteCols.Lat.name
    colLng = const.BusMapRouteCols.Lng.name
    colStationOrder = const.BusMapRouteCols.StationOrder.name
    colStationDir = const.BusMapRouteCols.StationDirection.name
    colRouteId = const.BusMapRouteCols.RouteId.name
    colPathPoints = const.BusMapRouteCols.pathPoints.name

    for subRoute in dfRawRoute[colStationDir].unique():
        dfSubRoute = dfRawRoute.loc[dfRawRoute[colStationDir] == subRoute]
        dfSubRoute = dfSubRoute.set_index(colStationOrder)
        dfSubRoute = dfSubRoute.sort_index()

        routeDict = {const.RoutesTtbCols.StationOrder.name: [],
                     const.RoutesTtbCols.StationCode.name: []}

        tripDist = 0
        routeID = -1
        stationDir = -1
        # goes through each station in this sorted subroute
        isFirstStop = True
        for index, row in dfSubRoute.iterrows():
            crnStnId = row[colStationId]

            # adds this node to routeDict
            routeDict[const.RoutesTtbCols.StationOrder.name].append(index)
            routeDict[const.RoutesTtbCols.StationCode.name].append(crnStnId)

            # adds this node to Graph ==================================================================================
            if crnStnId not in G.nodes or (crnStnId in G.nodes and not G.nodes[crnStnId]):
                nodeRoutes = pd.DataFrame.from_dict({const.GNodeRouteCols.RouteId.name: [row[colRouteId]],
                                                     const.GNodeRouteCols.StationDirection.name: [row[colStationDir]],
                                                     const.GNodeRouteCols.StationOrder.name: [index]})
                G.add_node(crnStnId)
                G.nodes[crnStnId][const.GNodeAttribs.routes.name] = nodeRoutes
                G.nodes[crnStnId][const.GNodeAttribs.StationId.name] = row[colStationId]
                G.nodes[crnStnId][const.GNodeAttribs.StationDesc.name] = row[colStationName] + ' - ' + row[colAddress]
                G.nodes[crnStnId][const.GNodeAttribs.Lat.name] = row[colLat]
                G.nodes[crnStnId][const.GNodeAttribs.Lng.name] = row[colLng]
            else:
                nodeRoutes = G.nodes[crnStnId][const.GNodeAttribs.routes.name]
                nodeRoutes.loc[len(nodeRoutes.index)] = [row[colRouteId], row[colStationDir], index]
                G.nodes[crnStnId][const.GNodeAttribs.routes.name] = nodeRoutes
            # ==========================================================================================================

            # adds this edge to graph G ================================================================================
            if isFirstStop:
                isFirstStop = False
            else:
                prevStnId = dfSubRoute.iloc[index - 1][colStationId]
                pathPointsStr = row[colPathPoints].split(' ')
                if row[colPathPoints]=='':
                    print('pathpoints empty')
                    pathPointsFloat = [ [dfSubRoute.iloc[index-1][const.BusMapRouteCols.Lng.name],dfSubRoute.iloc[index-1][const.BusMapRouteCols.Lat.name]],
                                        [row[const.BusMapRouteCols.Lng.name],row[const.BusMapRouteCols.Lat.name]]]
                else:
                    pathPointsFloat = [ [float(coord) for coord in pStr.split(',')] for pStr in pathPointsStr ]
                roadDist = utils.calcRoadDist(pathPointsFloat)
                tripDist += roadDist
                if G.has_edge(prevStnId, crnStnId):
                    edgeRoutes = G.get_edge_data(prevStnId, crnStnId)[const.GEdgeAttribs.routes.name]
                    crnIdx = len(edgeRoutes.index)
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.RouteId.name] = row[colRouteId]
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.StationDirection.name] = row[colStationDir]
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.edgeOrder.name] = index-1
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.pathPoints.name] = pathPointsFloat
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.roadDist.name] = roadDist
                    edgeRoutes.at[crnIdx, const.GEdgeRouteCols.tTrav.name] = -1

                    G.edges[prevStnId, crnStnId][const.GEdgeAttribs.routes.name] = edgeRoutes
                else:
                    edgeRoutes = pd.DataFrame.from_dict({const.GEdgeRouteCols.RouteId.name: [row[colRouteId]],
                                                         const.GEdgeRouteCols.StationDirection.name: [row[colStationDir]],
                                                         const.GEdgeRouteCols.edgeOrder.name: [index - 1],
                                                         const.GEdgeRouteCols.pathPoints.name: [pathPointsFloat],
                                                         const.GEdgeRouteCols.roadDist.name: [roadDist],
                                                         const.GEdgeRouteCols.tTrav.name: [-1]})
                    G.add_edge(prevStnId, crnStnId)
                    G.edges[prevStnId, crnStnId][const.GEdgeAttribs.routes.name] = edgeRoutes

            # ==========================================================================================================
            routeID = row[colRouteId]
            stationDir = row[colStationDir]

        # updates dfRoutes
        crnIdx = len(dfRoutes.index)
        dfRoutes.at[crnIdx, const.RoutesCols.RouteId.name] = routeID
        dfRoutes.at[crnIdx, const.RoutesCols.StationDirection.name] = stationDir
        dfRoutes.at[crnIdx, const.RoutesCols.tripDist.name] = tripDist
        dfRoutes.at[crnIdx, const.RoutesCols.designedTtb.name] = routeDict
        dfRoutes.at[crnIdx, const.RoutesCols.actualTtb.name] = routeDict


# ======================================================================================================================
def makeNetwork(jsFiles):
    G = nx.DiGraph()
    dfRoutes = pd.DataFrame(columns=utils.makeListFromEnum(const.RoutesCols))

    for jsFile in jsFiles:
        jsFilename = jsFile.split('/')[-1]
        if jsFilename in const.excludedRoutes:
            print('makeNetwork - reading in %s - EXCLLUDED' % jsFile)
            continue

        print('makeNetwork - reading in %s ...' % jsFile)
        with open(jsFile, 'r', encoding='utf-8') as f:
            jsRoute = json.load(f)

        dfRawRoute = pd.DataFrame(columns=utils.makeListFromEnum(const.BusMapRouteCols))
        for i in range(len(jsRoute)):
            dfRawRoute = utils.append2DF(dfRawRoute, jsRoute[i])

        fullFname = jsFile.split('/')[-1]
        fname = fullFname.split('.')[0]
        dfRawRoute.to_csv('%s/dfRaw%s.csv' % (const.trashSite, fname), index=False)

        # constructs graph G. Details of nodes and edges are from dfRawRoute
        addNodesEdgesToGraph(G, dfRoutes, dfRawRoute)

    return G, dfRoutes

