import networkx as nx
import pandas as pd
import os
import pickle

import constants as const
import utilities as utils

def addEdgesAlongRoutes_v2(GtopoTemp, dfRoutes):
    nodeDetails = []
    ndIDCounts = 0
    ndDesc = {}
    ndIDict = {}
    for idx, route in dfRoutes.iterrows():
        routeId = route[const.RoutesCols.RouteId.name]
        routeDir = route[const.RoutesCols.StationDirection.name]
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        ttb = designedTtb[const.RoutesTtbCols.timetable.name]

        print('addEdgesAlongRoutes route %d_%d, %s' % (routeId, routeDir, route[const.RoutesCols.RouteNo.name]))

        for service in ttb:
            # first stop
            iStn = 0
            crnArrTime = int(service[iStn][0])
            nxtArrTime = int(service[iStn + 1][0])
            # uses arrival time to create a node representing the 1st stop of a route.
            # departure time, which can be 6s after the arrival time, is not used.
            ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)
            if ndCrnArr not in ndDesc:
                ndIDCounts += 1
                ndDesc[ndCrnArr] = ndIDCounts
                ndIDict[ndIDCounts] = ndCrnArr
                nodeDetails.append([routeId, routeDir, stnList[iStn], crnArrTime,
                                    '%d_%d' % (routeId, routeDir), 'dep', ndIDCounts])

            ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
            if ndNxtArr not in ndDesc:
                ndIDCounts += 1
                ndDesc[ndNxtArr] = ndIDCounts
                ndIDict[ndIDCounts] = ndNxtArr
                nodeDetails.append([routeId, routeDir, stnList[iStn + 1], nxtArrTime,
                                    '%d_%d' % (routeId, routeDir), 'arr', ndIDCounts])

            GtopoTemp.add_edge(ndDesc[ndCrnArr], ndDesc[ndNxtArr])
            GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndNxtArr]][const.GttEdgeAttribs.type.name] = \
                const.GttEdgeAttribs.typeBus.value
            GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndNxtArr]][const.GttEdgeAttribs.dirRouteID.name] = \
                '%d_%d' % (routeId, routeDir)
            GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndNxtArr]][const.GttEdgeAttribs.time.name] = \
                nxtArrTime - crnArrTime
            GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndNxtArr]][const.GttEdgeAttribs.transfer.name] = 0

            # for each stop on this route except the last stop which we will only use the arrival time
            for iStn in range(1, len(service) - 1):
                crnArrTime = int(service[iStn][0])
                crnDepTime = int(service[iStn][1])
                nxtArrTime = int(service[iStn + 1][0])

                ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)

                ndCrnDep = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnDepTime)
                if ndCrnDep not in ndDesc:
                    ndIDCounts += 1
                    ndDesc[ndCrnDep] = ndIDCounts
                    ndIDict[ndIDCounts] = ndCrnDep
                    nodeDetails.append([routeId, routeDir, stnList[iStn], crnDepTime,
                                        '%d_%d' % (routeId, routeDir), 'dep', ndIDCounts])

                ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
                if ndNxtArr not in ndDesc:
                    ndIDCounts += 1
                    ndDesc[ndNxtArr] = ndIDCounts
                    ndIDict[ndIDCounts] = ndNxtArr
                    nodeDetails.append([routeId, routeDir, stnList[iStn + 1], nxtArrTime,
                                        '%d_%d' % (routeId, routeDir), 'arr', ndIDCounts])

                # print('%s, %s, %s' % (ndCrnArr, ndCrnDep, ndNxtArr))
                GtopoTemp.add_edge(ndDesc[ndCrnArr], ndDesc[ndCrnDep])
                GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndCrnDep]][const.GttEdgeAttribs.type.name] = \
                    const.GttEdgeAttribs.typeBus.value
                GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndCrnDep]][const.GttEdgeAttribs.dirRouteID.name] = \
                    '%d_%d' % (routeId, routeDir)
                GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndCrnDep]][const.GttEdgeAttribs.time.name] = \
                    crnDepTime - crnArrTime
                GtopoTemp.edges[ndDesc[ndCrnArr], ndDesc[ndCrnDep]][const.GttEdgeAttribs.transfer.name] = 0

                GtopoTemp.add_edge(ndDesc[ndCrnDep], ndDesc[ndNxtArr])
                GtopoTemp.edges[ndDesc[ndCrnDep], ndDesc[ndNxtArr]][const.GttEdgeAttribs.type.name] = \
                    const.GttEdgeAttribs.typeBus.value
                GtopoTemp.edges[ndDesc[ndCrnDep], ndDesc[ndNxtArr]][const.GttEdgeAttribs.dirRouteID.name] = \
                    '%d_%d' % (routeId, routeDir)
                GtopoTemp.edges[ndDesc[ndCrnDep], ndDesc[ndNxtArr]][const.GttEdgeAttribs.time.name] = \
                    nxtArrTime - crnDepTime
                GtopoTemp.edges[ndDesc[ndCrnDep], ndDesc[ndNxtArr]][const.GttEdgeAttribs.transfer.name] = 0

    dfNdDetails = pd.DataFrame(nodeDetails, columns=[const.dfNdDetailsCols.RouteId.name,
                                                     const.dfNdDetailsCols.StationDirection.name,
                                                     const.dfNdDetailsCols.StationId.name,
                                                     const.dfNdDetailsCols.time.name,
                                                     const.dfNdDetailsCols.dirRoute.name,
                                                     const.dfNdDetailsCols.arrOrDep.name,
                                                     const.dfNdDetailsCols.gttNdID.name])
    return dfNdDetails, ndIDict

# ======================================================================================================================
def makeTopoTempGraph_v2(dfRoutes):
    #  initiates a digraph
    GtopoTemp = nx.DiGraph()

    # constructs edges connecting consecutive stops for each service on each directed route
    GttTransitPkl = '%s/GttTransit.pkl' % const.picklesFolder
    dfNdDetailsPkl = '%s/dfNdDetails.pkl' % const.picklesFolder
    ndIDictPkl = '%s/ndIDict.pkl' % const.picklesFolder
    if os.path.isfile(GttTransitPkl) & os.path.isfile(dfNdDetailsPkl):
        GtopoTemp = pickle.load(open(GttTransitPkl, 'rb'))
        dfNdDetails = pickle.load(open(dfNdDetailsPkl, 'rb'))
        ndIDict = pickle.load(open(ndIDictPkl, 'rb'))
        print('unpickling GttTransitPkl and dfNdDetailsPkl completed')
    else:
        dfNdDetails, ndIDict = addEdgesAlongRoutes_v2(GtopoTemp, dfRoutes)
        with open(GttTransitPkl, 'wb') as f:
            pickle.dump(GtopoTemp, f)
        with open(dfNdDetailsPkl, 'wb') as f:
            pickle.dump(dfNdDetails, f)
        with open(ndIDictPkl, 'wb') as f:
            pickle.dump(ndIDict, f)
    print('finished addEdgesAlongRoutes')

    return GtopoTemp, dfNdDetails, ndIDict



# ======================================================================================================================
def addEdgesAlongRoutes(GtopoTemp, dfRoutes):
    nodeDetails = []

    for idx, route in dfRoutes.iterrows():
        routeId = route[const.RoutesCols.RouteId.name]
        routeDir = route[const.RoutesCols.StationDirection.name]
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        ttb = designedTtb[const.RoutesTtbCols.timetable.name]

        print('addEdgesAlongRoutes route %d_%d, %s' % (routeId, routeDir, route[const.RoutesCols.RouteNo.name]))
        # print(len(stnList))    # 33
        # print(len(ttb))         # 120, i.e. number of services
        # print(len(ttb[0]))      # 33, i.e. number of stations
        # print(ttb[0])           # [(18000, 18006), (18176.0, 18182.0), (18239.0, 18245.0), ..., ]

        for service in ttb:
            # first stop
            iStn = 0
            crnArrTime = int(service[iStn][0])
            nxtArrTime = int(service[iStn + 1][0])
            # uses arrival time to create a node representing the 1st stop of a route.
            # departure time, which can be 6s after the arrival time, is not used.
            ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)
            ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
            GtopoTemp.add_edge(ndCrnArr, ndNxtArr)
            GtopoTemp.edges[ndCrnArr, ndNxtArr][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
            GtopoTemp.edges[ndCrnArr, ndNxtArr][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId, routeDir)

            nodeDetails.append([routeId, routeDir, stnList[iStn], crnArrTime, '%d_%d' % (routeId, routeDir), 'dep'])
            nodeDetails.append([routeId, routeDir, stnList[iStn+1], nxtArrTime, '%d_%d' % (routeId, routeDir), 'arr'])

            # for each stop on this route except the last stop which we will only use the arrival time
            for iStn in range(1, len(service) - 1):
                crnArrTime = int(service[iStn][0])
                crnDepTime = int(service[iStn][1])
                nxtArrTime = int(service[iStn + 1][0])
                ndCrnArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnArrTime)
                ndCrnDep = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn], crnDepTime)
                ndNxtArr = '%d_%d_%d_%d' % (routeId, routeDir, stnList[iStn + 1], nxtArrTime)
                # print('%s, %s, %s' % (ndCrnArr, ndCrnDep, ndNxtArr))
                GtopoTemp.add_edge(ndCrnArr, ndCrnDep)
                GtopoTemp.edges[ndCrnArr, ndCrnDep][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
                GtopoTemp.edges[ndCrnArr, ndCrnDep][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId,routeDir)

                GtopoTemp.add_edge(ndCrnDep, ndNxtArr)
                GtopoTemp.edges[ndCrnDep, ndNxtArr][const.GttEdgeAttribs.type.name] = const.GttEdgeAttribs.typeBus.value
                GtopoTemp.edges[ndCrnDep, ndNxtArr][const.GttEdgeAttribs.dirRouteID.name] = '%d_%d' % (routeId,routeDir)

                nodeDetails.append([routeId, routeDir, stnList[iStn], crnDepTime,
                                    '%d_%d' % (routeId, routeDir), 'dep'])
                nodeDetails.append([routeId, routeDir, stnList[iStn + 1], nxtArrTime,
                                    '%d_%d' % (routeId, routeDir), 'arr'])

    return pd.DataFrame(nodeDetails, columns=[const.dfNdDetailsCols.RouteId.name,
                                              const.dfNdDetailsCols.StationDirection.name,
                                              const.dfNdDetailsCols.StationId.name,
                                              const.dfNdDetailsCols.time.name,
                                              const.dfNdDetailsCols.dirRoute.name,
                                              const.dfNdDetailsCols.arrOrDep.name])

# ======================================================================================================================
def makeTopoTempGraph(dfRoutes):
    #  initiates a digraph
    GtopoTemp = nx.DiGraph()

    # constructs edges connecting consecutive stops for each service on each directed route
    GttTransitPkl = '%s/GttTransit.pkl' % const.picklesFolder
    dfNdDetailsPkl = '%s/dfNdDetails.pkl' % const.picklesFolder
    if os.path.isfile(GttTransitPkl) & os.path.isfile(dfNdDetailsPkl):
        GtopoTemp = pickle.load(open(GttTransitPkl, 'rb'))
        dfNdDetails = pickle.load(open(dfNdDetailsPkl, 'rb'))
        print('unpickling GttTransitPkl and dfNdDetailsPkl completed')
    else:
        dfNdDetails = addEdgesAlongRoutes(GtopoTemp, dfRoutes)
        with open(GttTransitPkl, 'wb') as f:
            pickle.dump(GtopoTemp, f)
        with open(dfNdDetailsPkl, 'wb') as f:
            pickle.dump(dfNdDetails, f)
    print('finished addEdgesAlongRoutes')

    return GtopoTemp, dfNdDetails