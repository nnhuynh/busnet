import pandas as pd
import igraph
import os
import pickle
import random
import multiprocessing
import time

import constants as const
import ttgVisual_v2

def testAddVertices(gtt):
    for i in range(7):
        gtt.add_vertex(i)

def testAddEdges(gtt):
    gtt.add_edges([(0, 1), (1, 2), (2, 3), (3, 6),
                   (0, 4), (4, 3), (4, 2), (2, 5),
                   (0, 5), (6, 5)])

def testSite():
    gtt = igraph.Graph(directed=True)

    testAddVertices(gtt)
    testAddEdges(gtt)

    # assigns vertice attribute
    gtt.vs['name'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    # adds a single vertex
    gtt.add_vertex(7)
    gtt.vs[7]['name'] = 'h'
    # print vertex
    for i in range(len(gtt.vs)):
        print('%d, %s' % (i, gtt.vs[i]['name']))
    print(gtt.vs['name'])

    # assigns edge attribute
    gtt.es[gtt.get_eid(0, 1)]['transfer'] = 1
    gtt.es[gtt.get_eid(1, 2)]['transfer'] = 0
    gtt.es[gtt.get_eid(2, 3)]['transfer'] = 0
    gtt.es[gtt.get_eid(3, 6)]['transfer'] = 0
    gtt.es[gtt.get_eid(0, 4)]['transfer'] = 1
    gtt.es[gtt.get_eid(4, 3)]['transfer'] = 0
    gtt.es[gtt.get_eid(4, 2)]['transfer'] = 1
    gtt.es[gtt.get_eid(2, 5)]['transfer'] = 1
    gtt.es[gtt.get_eid(0, 5)]['transfer'] = 2
    gtt.es[gtt.get_eid(6, 5)]['transfer'] = 0

    gtt.es[gtt.get_eid(0, 1)]['dirRoute'] = '10_1'
    gtt.es[gtt.get_eid(1, 2)]['dirRoute'] = '10_1'
    gtt.es[gtt.get_eid(2, 3)]['dirRoute'] = '10_1'
    gtt.es[gtt.get_eid(3, 6)]['dirRoute'] = '10_0'
    gtt.es[gtt.get_eid(0, 4)]['dirRoute'] = '10_0'
    gtt.es[gtt.get_eid(4, 3)]['dirRoute'] = '10_0'
    gtt.es[gtt.get_eid(4, 2)]['dirRoute'] = '10_1'
    gtt.es[gtt.get_eid(2, 5)]['dirRoute'] = '10_1'
    gtt.es[gtt.get_eid(0, 5)]['dirRoute'] = '10_0'
    gtt.es[gtt.get_eid(6, 5)]['dirRoute'] = '10_1'

    print(gtt.es['dirRoute'])
    print(type(gtt.es['dirRoute']))
    edgeSet = gtt.es.select(dirRoute_in=['10_1'])
    edgeSubset = edgeSet.select(transfer_eq=1)
    for edge in edgeSubset:
        print([edge.source, edge.source_vertex['name'],
               edge.target, edge.target_vertex['name'],
               edge['dirRoute'], edge['transfer']])

    if 1 in [v.index for v in gtt.vs]:
        print('1 is in gtt.vs')

    if 10 in [v.index for v in gtt.vs]:
        print('10 is in gtt.vs')
    else:
        print('10 is NOT in gtt.vs')

    print('\nShortest path results')
    paths_af = igraph.GraphBase.get_shortest_paths(gtt, v=6, to=[4,5,1], mode='OUT', weights='transfer')
    print(paths_af)
    print(paths_af[0])
    print(type(paths_af[0]))
    print(len(paths_af[0]))
    if len(paths_af[0])>0:
        print('path found')
    else:
        print('path not found')

    '''
    # add new edges
    print('\nnew edge')
    gtt.add_edges([(4, 6), (4, 7), (5, 7)])
    print(gtt.es['transfer'])

    newEdges = gtt.es.select(transfer_in=[None])
    newEdges['transfer'] = 1
    for edge in newEdges:
        print('(%d, %d, %d)' % (edge.source, edge.target, edge['transfer']))

    print('after')
    print(gtt.es['transfer'])
    '''

# ======================================================================================================================
def addTransitEdges(dfRoutes):
    gtt = igraph.Graph(directed = True)
    nodeDetails = []
    ndIDCounts = -1
    ndDesc = {}
    ndIDict = {}
    countRoutes = 0
    for idx, route in dfRoutes.iterrows():
        countRoutes += 1
        routeId = route[const.RoutesCols.RouteId.name]
        routeDir = route[const.RoutesCols.StationDirection.name]
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        ttb = designedTtb[const.RoutesTtbCols.timetable.name]

        for service in ttb:
            # first stop
            iStn = 0
            crnArrTime = int(service[iStn][0])
            nxtArrTime = int(service[iStn + 1][0])

            # uses arrival time to create a node representing the 1st stop of a route.
            # departure time, which can be 6s after the arrival time, is not used.
            ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)
            ndIDCounts += 1
            # adds 1 new vertex to gtt
            gtt.add_vertex(ndIDCounts)
            gtt.vs[ndIDCounts]['ndDesc'] = ndCrnArr
            # updates other variables
            ndDesc[ndCrnArr] = ndIDCounts
            ndIDict[ndIDCounts] = ndCrnArr
            nodeDetails.append([routeId, routeDir, stnList[iStn], crnArrTime, '%d_%d' % (routeId, routeDir),
                                'dep', ndIDCounts])

            ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
            ndIDCounts += 1
            # adds 1 new vertex to gtt
            gtt.add_vertex(ndIDCounts)
            gtt.vs[ndIDCounts]['ndDesc'] = ndNxtArr
            # updates other variables
            ndDesc[ndNxtArr] = ndIDCounts
            ndIDict[ndIDCounts] = ndNxtArr
            nodeDetails.append([routeId, routeDir, stnList[iStn + 1], nxtArrTime, '%d_%d' % (routeId, routeDir),
                                'arr', ndIDCounts])

            # adds 1 new edge to gtt
            gtt.add_edge(ndDesc[ndCrnArr], ndDesc[ndNxtArr])
            eid = gtt.get_eid(ndDesc[ndCrnArr], ndDesc[ndNxtArr])
            gtt.es[eid][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
            gtt.es[eid][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId, routeDir)
            gtt.es[eid][const.GttEdgeAttribs.time.name] = nxtArrTime - crnArrTime
            gtt.es[eid][const.GttEdgeAttribs.transfer.name] = 0

            #  for each stop on this route except the last stop which we will only use the arrival time
            for iStn in range(1, len(service) - 1):
                crnArrTime = int(service[iStn][0])
                crnDepTime = int(service[iStn][1])
                nxtArrTime = int(service[iStn + 1][0])

                ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)

                ndCrnDep = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnDepTime)
                ndIDCounts += 1
                # add 1 new vertex to gtt
                gtt.add_vertex(ndIDCounts)
                gtt.vs[ndIDCounts]['ndDesc'] = ndCrnDep
                # update relevent variables
                ndDesc[ndCrnDep] = ndIDCounts
                ndIDict[ndIDCounts] = ndCrnDep
                nodeDetails.append([routeId, routeDir, stnList[iStn], crnDepTime, '%d_%d' % (routeId, routeDir),
                                    'dep', ndIDCounts])

                ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
                ndIDCounts += 1
                # add 1 new vertex to gtt
                gtt.add_vertex(ndIDCounts)
                gtt.vs[ndIDCounts][const.GttNodeAttribs.ndDesc.name] = ndNxtArr
                # update relevant variables
                ndDesc[ndNxtArr] = ndIDCounts
                ndIDict[ndIDCounts] = ndNxtArr
                nodeDetails.append([routeId, routeDir, stnList[iStn + 1], nxtArrTime, '%d_%d' % (routeId, routeDir),
                                    'arr', ndIDCounts])

                # adds 1 new edge to gtt for dwelling at the current stop
                gtt.add_edge(ndDesc[ndCrnArr], ndDesc[ndCrnDep])
                eid = gtt.get_eid(ndDesc[ndCrnArr], ndDesc[ndCrnDep])
                gtt.es[eid][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
                gtt.es[eid][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId, routeDir)
                gtt.es[eid][const.GttEdgeAttribs.time.name] = crnDepTime - crnArrTime
                gtt.es[eid][const.GttEdgeAttribs.transfer.name] = 0

                # adds 1 new edge to gtt
                gtt.add_edge(ndDesc[ndCrnDep], ndDesc[ndNxtArr])
                eid = gtt.get_eid(ndDesc[ndCrnDep], ndDesc[ndNxtArr])
                gtt.es[eid][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
                gtt.es[eid][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId, routeDir)
                gtt.es[eid][const.GttEdgeAttribs.time.name] = nxtArrTime - crnDepTime
                gtt.es[eid][const.GttEdgeAttribs.transfer.name] = 0

        if countRoutes % 10 == 0:
            print('%d routes added to gtt out of %d total routes' % (countRoutes, dfRoutes.shape[0]))

    dfNdDetails = pd.DataFrame(nodeDetails, columns=[const.dfNdDetailsCols.RouteId.name,
                                                     const.dfNdDetailsCols.StationDirection.name,
                                                     const.dfNdDetailsCols.StationId.name,
                                                     const.dfNdDetailsCols.time.name,
                                                     const.dfNdDetailsCols.dirRoute.name,
                                                     const.dfNdDetailsCols.arrOrDep.name,
                                                     const.dfNdDetailsCols.gttNdID.name])

    return gtt, dfNdDetails, ndIDict, ndDesc

# ======================================================================================================================
def mkTransferEdgesSubset(stopSubset, allStops, dfNdDetails, stopPairsDist, timeWindow, filename):
    transferEdges = pd.DataFrame()
    countStopFr = 0
    for stopFr in stopSubset:
        dfStopFrArrNds = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
                                         (dfNdDetails[const.dfNdDetailsCols.time.name] >= timeWindow[0]) &
                                         (dfNdDetails[const.dfNdDetailsCols.time.name] <= timeWindow[1]) &
                                         (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                          const.dfNdDetailsCols.arr.name)]
        # makes wait edges
        uDirRoutes = dfStopFrArrNds[const.dfNdDetailsCols.dirRoute.name].unique()
        if uDirRoutes.shape[0] > 1: # we only adds wait edges between stops on different routes
            for idxFr, rowFr in dfStopFrArrNds.iterrows():
                nodeFr = rowFr[const.dfNdDetailsCols.gttNdID.name]
                dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
                timeFr = rowFr[const.dfNdDetailsCols.time.name]
                dfStopFrDepNds = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopFr) &
                                                 (dfNdDetails[const.dfNdDetailsCols.time.name] > timeFr) &
                                                 (dfNdDetails[const.dfNdDetailsCols.time.name] <= min(
                                                     timeFr + const.maxWaitTime, timeWindow[1])) &
                                                 (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                                 (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                  const.dfNdDetailsCols.dep.name)]
                dfStopFrDepNds = dfStopFrDepNds.assign(nodeFr = nodeFr)
                dfStopFrDepNds = dfStopFrDepNds.assign(type = const.GttEdgeAttribs.typeWait.value)
                transferEdges = pd.concat([transferEdges,
                                           dfStopFrDepNds[['nodeFr', const.dfNdDetailsCols.gttNdID.name, 'type']]])

        # makes walk edges
        for stopTo in allStops:
            if stopFr == stopTo: continue
            dist = stopPairsDist['%d_%d' % (stopFr, stopTo)]
            if dist > const.maxWalkDist: continue

            walkTime = int(dist / const.walkSpeed + .5)  # in seconds

            for idxFr, rowFr in dfStopFrArrNds.iterrows():
                nodeFr = rowFr[const.dfNdDetailsCols.gttNdID.name]
                dirRouteFr = rowFr[const.dfNdDetailsCols.dirRoute.name]
                timeFr = rowFr[const.dfNdDetailsCols.time.name]
                dfStopToDepNds = dfNdDetails.loc[(dfNdDetails[const.dfNdDetailsCols.StationId.name] == stopTo) &
                                                 (dfNdDetails[const.dfNdDetailsCols.time.name] >= timeFr + walkTime) &
                                                 (dfNdDetails[const.dfNdDetailsCols.time.name] <=
                                                  min(timeFr + walkTime + const.maxWaitTime, timeWindow[1])) &
                                                 (dfNdDetails[const.dfNdDetailsCols.dirRoute.name] != dirRouteFr) &
                                                 (dfNdDetails[const.dfNdDetailsCols.arrOrDep.name] ==
                                                  const.dfNdDetailsCols.dep.name)]
                dfStopToDepNds = dfStopToDepNds.assign(nodeFr = nodeFr)
                dfStopToDepNds = dfStopToDepNds.assign(type = const.GttEdgeAttribs.typeWalk.value)
                transferEdges = pd.concat([transferEdges,
                                           dfStopToDepNds[['nodeFr', const.dfNdDetailsCols.gttNdID.name, 'type']]])

    countStopFr += 1
    if countStopFr % 10 == 0:
        print('mkTransferEdgesSubset for %d stopFr completed (out of %d stops)' % (countStopFr, len(stopSubset)))

    transferEdges.to_csv(filename, index=False)

# ======================================================================================================================
def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# ======================================================================================================================
def mkTransferEdgesMultip(G, dfNdDetails, timeWindow, stopPairsDist, nProcesses, transfersEdgeFilePrefix):
    allStops = list(G.nodes)
    # randomly split GttSubStops into nProcesses parts
    stopsParts = partition(allStops, nProcesses)

    # creates parallel processes to create transfer (i.e. wait and walk) edges
    mkTransferProcesses = []
    for i in range(nProcesses):
        stopSubset = stopsParts[i]
        p = multiprocessing.Process(target = mkTransferEdgesSubset,
                                    args = (stopSubset, allStops, dfNdDetails, stopPairsDist, timeWindow,
                                            '%s_%d.csv' % (transfersEdgeFilePrefix, i)))
        #mkTransferEdgesSubset(stopSubset, allStops, dfNdDetails, stopPairsDist, timeWindow, filename)
        mkTransferProcesses.append(p)
        p.start()
        print('mkTransferEdgesMultip process %d started - %d nodes' % (i, len(stopSubset)))

    for p in mkTransferProcesses:
        p.join()


# ======================================================================================================================
def getStop(node, ndIDict):
    return int(ndIDict[node].split('_')[2])

def getTime(node, ndIDict):
    return int(ndIDict[node].split('_')[3])

def consolidateTransferEdges(transferEdgesFiles, ndIDict):
    # construct a dataframe with columns: stopFr, stopTo, nodeFr, nodeTo, timeFr, timeTo, type
    allTransferEdges = pd.DataFrame()
    for file in transferEdgesFiles:
        starttime = time.perf_counter()
        print('reading file %s...' % file)
        dfTransfer = pd.read_csv(file)

        dfTransfer['stopFr'] = dfTransfer['nodeFr'].apply(getStop, ndIDict=ndIDict)
        dfTransfer['stopTo'] = dfTransfer[const.dfNdDetailsCols.gttNdID.name].apply(getStop, ndIDict=ndIDict)
        dfTransfer['timeFr'] = dfTransfer['nodeFr'].apply(getTime, ndIDict=ndIDict)
        dfTransfer['timeTo'] = dfTransfer[const.dfNdDetailsCols.gttNdID.name].apply(getTime, ndIDict=ndIDict)
        dfTransfer = dfTransfer.rename(columns={const.dfNdDetailsCols.gttNdID.name: 'nodeTo'})

        allTransferEdges = pd.concat([allTransferEdges, dfTransfer])
        duration = time.perf_counter() - starttime
        print('\tfinished in %.2g seconds (%.2g minutes)' % (duration, duration/60))

    return allTransferEdges

# ======================================================================================================================
def addTransferEdges(gttTransit, transferEdgesFiles, ndIDict):
    #vs = [v.index for v in gttTransit.vs]
    for file in transferEdgesFiles:
        print('reading file %s...' % file)
        dfTransferEdges = pd.read_csv(file)
        countRows = 0
        for idx, row in dfTransferEdges.iterrows():
            nodeFr = row['nodeFr']
            timeFr = int(ndIDict[nodeFr].split('_')[3])
            nodeTo = row[const.dfNdDetailsCols.gttNdID.name]
            timeTo = int(ndIDict[nodeTo].split('_')[3])
            #if nodeFr not in vs:
            #    print('WARNING: node %d not in GttSub' % nodeFr)
            #if nodeTo not in vs:
            #    print('WARNING: node %d not in GttSub' % nodeTo)
            gttTransit.add_edge(nodeFr, nodeTo)
            eid = gttTransit.get_eid(nodeFr, nodeTo)
            gttTransit.es[eid][const.GttEdgeAttribs.type.name] = row['type']
            gttTransit.es[eid][const.GttEdgeAttribs.time.name] = timeTo - timeFr
            gttTransit.es[eid][const.GttEdgeAttribs.transfer.name] = 1
            gttTransit.es[eid][const.GttEdgeAttribs.dirRouteID.name] = ''

            countRows += 1
            if countRows % 1e5 == 0:
                print('\tcompleted %d rows (out of %d)' % (countRows, dfTransferEdges.shape[0]))

# ======================================================================================================================
def ttgTest(gttComplete, dfNdDetails, ndIDict):
    # TEST 1: print and plot transit edges belonging to a directed route in a time window
    dirRoute = '10_0'
    startHr = 5
    endHr = 5.5
    timeWindow = [startHr*3600, endHr*3600] # from 5am to 6am
    # gets edges that have dirRoute
    edgeSet = gttComplete.es.select(dirRouteID_in = [dirRoute])
    edgeSubset = []

    for edge in edgeSet:
        mismatchFound = False
        sourceNdID = edge.source
        sourceNdDesc = edge.source_vertex['ndDesc']
        if ndIDict[sourceNdID] != sourceNdDesc:
            mismatchFound = True
            print('mismatch source vertexID %d: %s in ndIDict, %s in vertex.ndDesc' %
                  (sourceNdID, ndIDict[sourceNdID], sourceNdDesc))

        targetNdID = edge.target
        targetNdDesc = edge.target_vertex['ndDesc']
        if ndIDict[targetNdID] != targetNdDesc:
            mismatchFound = True
            print('mismatch target vertexID %d, %s in ndIDict, %s in vertex.ndDesc' %
                  (sourceNdID, ndIDict[sourceNdID], sourceNdDesc))

        if mismatchFound: continue

        # gets the time at source and target
        sourceTime = int(sourceNdDesc.split('_')[3])
        targetTime = int(targetNdDesc.split('_')[3])
        if (sourceTime>=timeWindow[0] and sourceTime<=timeWindow[1]) or \
                (targetTime>=timeWindow[0] and targetTime<=timeWindow[1]):
            edgeSubset.append(edge)
            print('%d (%s) - %d (%s)' % (sourceNdID, sourceNdDesc, targetNdID, targetNdDesc))

    ttgVisual_v2.plotTransitLinks(gttComplete, edgeSubset, timeWindow, '%s_5_6.png' % dirRoute)

    # TEST 2: print all walk links FROM a given/random node
    # - if the node is a 'dep' node: there should be no links
    # - if the node is an 'arr' node: all links should have walk type, the link time should be <= 64, stops should be <= 300m apart
    # randomly picks a vertex
    ndID = 1000
    edgeSet = gttComplete.es.select(type_in = [const.GttEdgeAttribs.typeWalk.value,
                                               const.GttEdgeAttribs.typeWalk.value])


# ======================================================================================================================
def makeTopoTempGraphMultip(G, dfRoutes, stopPairsDist, nProcesses):
    #  initiates a digraph
    # constructs edges connecting consecutive stops for each service on each directed route
    gttTransitPkl = '%s/GttTransit_v2.pkl' % const.picklesFolder
    dfNdDetailsPkl = '%s/dfNdDetails_v2.pkl' % const.picklesFolder
    ndIDictPkl = '%s/ndIDict_v2.pkl' % const.picklesFolder
    ndIDictInvPkl = '%s/ndIDictInv_v2.pkl' % const.picklesFolder
    dfAllTransfersPkl = '%s/dfAllTransfers.pkl' % const.picklesFolder

    if (os.path.isfile(gttTransitPkl)) and (os.path.isfile(dfNdDetailsPkl)):
        gttTransit = pickle.load(open(gttTransitPkl, 'rb'))
        dfNdDetails = pickle.load(open(dfNdDetailsPkl, 'rb'))
        ndIDict = pickle.load(open(ndIDictPkl, 'rb'))
        ndIDictInv = pickle.load(open(ndIDictInvPkl, 'rb'))
        print('unpickling GttTransitPkl and dfNdDetailsPkl completed')
    else:
        gttTransit, dfNdDetails, ndIDict, ndIDictInv = addTransitEdges(dfRoutes)
        with open(gttTransitPkl, 'wb') as f:
            pickle.dump(gttTransit, f)
        with open(dfNdDetailsPkl, 'wb') as f:
            pickle.dump(dfNdDetails, f)
        with open(ndIDictPkl, 'wb') as f:
            pickle.dump(ndIDict, f)
        with open(ndIDictInvPkl, 'wb') as f:
            pickle.dump(ndIDictInv, f)
        print('adding transit edges completed')


    if os.path.isfile(dfAllTransfersPkl):
        dfAllTransfers = pickle.load(open(dfAllTransfersPkl, 'rb'))
        print('unpickling dfAllTransfersPkl completed')
    else:
        # makes transfer edges with multiprocessing
        transfersEdgeFilePrefix = '%s/transferEdges' % const.trashSite
        timeWindow = [0, 24*3600]
        mkTransferEdgesMultip(G, dfNdDetails, timeWindow, stopPairsDist, nProcesses, transfersEdgeFilePrefix)
        print('finished mkTransferEdgesMultip')

        # consolidates wait edges and walk edges from all multiprocessing runs
        # puts them in a dataframe with columns ['nodeFr', 'nodeTo', 'type', 'stopFr', 'stopTo', 'timeFr', 'timeTo']
        # and pickles this dataframe
        transferEdgesFiles = ['%s_%d.csv' % (transfersEdgeFilePrefix, i) for i in range(nProcesses)]
        #addTransferEdges(gttTransit, transferEdgesFiles, ndIDict)  # DO NOT RUN addTransferEdges, IT TAKES FOREVER!!!
        dfAllTransfers = consolidateTransferEdges(transferEdgesFiles, ndIDict)

        with open(dfAllTransfersPkl, 'wb') as f:
            pickle.dump(dfAllTransfers, f)
        print('pickling transfer edges completed')

    return gttTransit, dfNdDetails, ndIDict, ndIDictInv, dfAllTransfers


