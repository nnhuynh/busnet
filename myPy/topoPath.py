import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import utilities as utils
import constants as const

# ======================================================================================================================
def calcShortestPaths_v3(G):
    '''
    to be run on BUDF only - will take ages on my laptop
    :param G:
    :return:
    '''
    shortHops = dict(nx.all_pairs_dijkstra(G))
    print('shortHops completed')
    shortDist = dict(nx.all_pairs_dijkstra(G, weight='meanRoadDist'))
    print('shortDist completed')
    shortTime = dict(nx.all_pairs_dijkstra(G, weight='meanTravTime'))
    print('shortTime completed')
    '''
    with open('./pickles/shortHops.pkl', 'wb') as f:
        pickle.dump(shortHops, f)
    with open('./pickles/shortDist.pkl', 'wb') as f:
        pickle.dump(shortDist, f)
    with open('./pickles/shortTime.pkl', 'wb') as f:
        pickle.dump(shortTime, f)
    '''
    return shortHops, shortDist, shortTime

# ======================================================================================================================
def convert2Str(intList):
    strVals = ''
    for val in intList:
        strVals = '%s_%s' % (strVals, str(val))
    return strVals

def convertToDataframe(shortHops, shortDist, shortTime):
    '''
    :param G:
    :param shortHops:
    :param shortDist:
    :param shortTime:
    :return:
    '''
    # converts shortHops (in dictionary) to dataframes dfShortHopsValue and dfShortHopsPath
    recordsValue = []
    recordsPath = []
    for nodei, valNodei in shortHops.items():
        for nodej, valNodej in valNodei[0].items():
            recordsValue.append((nodei, nodej, valNodej))
        for nodej, valNodej in valNodei[1].items():
            recordsPath.append((nodei, nodej, valNodej))

    dfShortHopsValue = pd.DataFrame(recordsValue, columns=['nodei', 'nodej', 'value'])
    dfShortHopsPath = pd.DataFrame(recordsPath, columns=['nodei', 'nodej', 'path'])
    print('finished dfShortHopsValue and dfShortHopsPath')

    # converts shortDist (in dictionary) to dataframes dfShortDistValue and dfShortDistPath
    recordsValue = []
    recordsPath = []
    for nodei, valNodei in shortDist.items():
        for nodej, valNodej in valNodei[0].items():
            recordsValue.append((nodei, nodej, valNodej))
        for nodej, valNodej in valNodei[1].items():
            recordsPath.append((nodei, nodej, valNodej))

    dfShortDistValue = pd.DataFrame(recordsValue, columns=['nodei', 'nodej', 'value'])
    dfShortDistPath = pd.DataFrame(recordsPath, columns=['nodei', 'nodej', 'path'])
    print('finished dfShortDistValue and dfShortDistPath')

    # converts shortTime (in dictionary) to dataframes dfShortTimeValue and dfShortTimePath
    recordsValue = []
    recordsPath = []
    for nodei, valNodei in shortTime.items():
        for nodej, valNodej in valNodei[0].items():
            recordsValue.append((nodei, nodej, valNodej))
        for nodej, valNodej in valNodei[1].items():
            recordsPath.append((nodei, nodej, valNodej))

    dfShortTimeValue = pd.DataFrame(recordsValue, columns=['nodei', 'nodej', 'value'])
    dfShortTimePath = pd.DataFrame(recordsPath, columns=['nodei', 'nodej', 'path'])
    print('finished dfShortTimeValue and dfShortTimePath')

    # adds column 'value' in dfShortHopsValue to dfShortHopsPath by matching nodei and nodej
    # repeats for dfShortDistPath and dfShortTimePath
    dfShortHops = pd.merge(dfShortHopsPath, dfShortHopsValue, how='left', left_on=['nodei', 'nodej'],
                           right_on=['nodei', 'nodej'])
    dfShortDist = pd.merge(dfShortDistPath, dfShortDistValue, how='left', left_on=['nodei', 'nodej'],
                           right_on=['nodei', 'nodej'])
    dfShortTime = pd.merge(dfShortTimePath, dfShortTimeValue, how='left', left_on=['nodei', 'nodej'],
                           right_on=['nodei', 'nodej'])

    # converts node list in column 'path' into string, e.g. [1, 3, 4, 654] to '_1_3_4_634'
    dfShortHops['pathStr'] = dfShortHops['path'].apply(convert2Str)
    print('complete pathStr for dfShortHops')

    dfShortDist['pathStr'] = dfShortDist['path'].apply(convert2Str)
    print('complete pathStr for dfShortDist')

    dfShortTime['pathStr'] = dfShortTime['path'].apply(convert2Str)
    print('complete pathStr for dfShortTime')

    #with open('./pickles/dfShortHopsPath.pkl', 'wb') as f:
    #    pickle.dump(dfShortHops, f)
    #with open('./pickles/dfShortDistPath.pkl', 'wb') as f:
    #    pickle.dump(dfShortDist, f)
    #with open('./pickles/dfShortTimePath.pkl', 'wb') as f:
    #    pickle.dump(dfShortTime, f)

    return dfShortHops, dfShortDist, dfShortTime


# ======================================================================================================================
def calcShortestPaths_v2(G, dfRoutes, edgeWeight=None):
    '''
    calculate shortest path between all node pairs in G, except for:
    - two identical nodes
    - two nodes within 300m of each other and not directly connected.
    - two nodes on the same directed route
    :param G:
    :param edgeWeight:
    :return:
    '''
    shortestPaths = {}
    for nodei in G.nodes():
        shortestPaths[nodei] = {}
        for nodej in G.nodes():
            # IF THE 2 NODES ARE IDENTICAL
            if nodej==nodei:
                continue

            # IF THERE'S A DIRECT LINK BETWEEN THEM
            if G.has_edge(nodei, nodej):
                shortestPaths[nodei][nodej] = {}
                shortestPaths[nodei][nodej]['path'] = [nodei, nodej]
                shortestPaths[nodei][nodej]['len'] = calcPathLength(G, [nodei, nodej], edgeWeight)
                continue

            # IF THEY ARE ON THE SAME DIRECTED ROUTE (AND nodei IS BEFORE nodej)
            # combines routeID_StationDirection (e.g. 25_0) of all directed routes passing nodei and passing nodej
            # rNdi and rNdj are pandas Series
            rNdi = G.nodes[nodei][const.GNodeAttribs.routes.name].apply(
                lambda row: '%d_%d' % (row[const.GNodeRouteCols.RouteId.name],
                                       row[const.GNodeRouteCols.StationDirection.name]), axis=1)
            rNdj = G.nodes[nodej][const.GNodeAttribs.routes.name].apply(
                lambda row: '%d_%d' % (row[const.GNodeRouteCols.RouteId.name],
                                       row[const.GNodeRouteCols.StationDirection.name]), axis=1)
            # if found at least one common directed route, i.e. nodei and nodej are on the same directed route
            commonRoutes = list(set(rNdi.values).intersection(set(rNdj.values)))
            # reverse engineer to get routeID and StationDirection using the 1st value in commonRoutes
            # ideally we should examine all routes in commonRoutes and pick the shortest.
            # However picking the 1st route that have nodei before nodej should be fine 95% of the time.
            # Note that it is possible that we may not find any such route in commonRoutes
            onSameRoute = False
            for route in commonRoutes:
                routeID = int(route.split('_')[0])
                stnDir = int(route.split('_')[1])
                ttb = dfRoutes['designedTtb'].loc[(dfRoutes['RouteId'] == routeID) &
                                                  (dfRoutes['StationDirection'] == stnDir)].values[0]
                idxNodei = ttb['StationCode'].index(nodei)
                idxNodej = ttb['StationCode'].index(nodej)
                if idxNodei < idxNodej:
                    onSameRoute = True
                    sPath = ttb['StationCode'][idxNodei:idxNodej+1]
                    shortestPaths[nodei][nodej] = {}
                    shortestPaths[nodei][nodej]['path'] = sPath
                    shortestPaths[nodei][nodej]['len'] = calcPathLength(G, sPath, edgeWeight)
                    break
            if onSameRoute:
                continue

            # IF 2 NODES ARE WITHIN WALK DISTANCE TO EACH OTHER AND ARE NOT ON THE SAME DIRECTED ROUTE
            nodeiCoord = [G.nodes[nodei][const.GNodeAttribs.Lng.name], G.nodes[nodei][const.GNodeAttribs.Lat.name]]
            nodejCoord = [G.nodes[nodej][const.GNodeAttribs.Lng.name], G.nodes[nodej][const.GNodeAttribs.Lat.name]]
            dist = utils.calcGeodesicDist(nodeiCoord, nodejCoord)  # distance in metres
            if dist <= const.maxWalkDist:
                continue

            # NONE OF THE ABOVE
            sPath = nx.shortest_path(G, source=nodei, target=nodej, weight=edgeWeight, method='dijkstra') # method='bellman-ford'
            shortestPaths[nodei][nodej] = {}
            shortestPaths[nodei][nodej]['path'] = sPath
            shortestPaths[nodei][nodej]['len'] = calcPathLength(G, sPath, edgeWeight)

    return shortestPaths

# ======================================================================================================================
def calcPathLength(G, path, edgeWeight=None):
    # the length of an unweighted path is the number of edges it traverses.
    if edgeWeight==None:
        return len(path)-1
    pathLen = 0
    for i in range(1,len(path)):
        preNode = path[i-1]
        crnNode = path[i]
        pathLen += G.edges[preNode,crnNode][edgeWeight]
    return pathLen



'''
# ======================================================================================================================
# ======================================================================================================================
The below functions are PRACTICALLY USELESS!!!
'''
# ======================================================================================================================
def calcPathLenDistrib(shortestPaths, description, xLabel, yLabel):
    '''
    :param shortestPaths:
    :param description: e.g. 'pathLen'
    :return:
    '''
    pathList = []
    for nodei in shortestPaths:
        nodejs = shortestPaths[nodei].keys()
        for nodej in nodejs:
            pathList.append([nodei, nodej, shortestPaths[nodei][nodej]['len']])
    dfPaths = pd.DataFrame(pathList, columns=['nodei', 'nodej', 'len'])

    probs = dfPaths['len'].value_counts(normalize=True)
    probs = probs.sort_index()
    # fits probablity of path lengths (number of hops) vs path lengths following the asymmetric unimodal function
    xObs = probs.index.to_list()
    yObs = probs.values
    result = utils.fitAsymetricUnimodal(xObs, yObs, initA=1, initB=1, initC=1)

    summaryFilename = '%s/%s_modelFitSummary.txt' % (const.pathLenDistribFolder, description)
    plotFilename = '%s/%s_plot.png' % (const.pathLenDistribFolder, description)

    # saves result summary to text file
    with open(summaryFilename,'w') as fh:
        fh.write(result.fit_report())

    # plots path lengths vs path length probability
    plt.plot(xObs, yObs, '.', color='black', label='raw data')
    plt.plot(xObs, result.best_fit, '-', color='red', label='best fist')
    plt.xlabel(xLabel, fontsize=11)
    plt.ylabel(yLabel, fontsize=11)
    plt.tight_layout()
    plt.savefig(plotFilename, bbox_inches='tight', pad_inches=0)
    plt.clf()

# ======================================================================================================================
def calcPathDistrib(shortestWeightedPaths, description, xLabel, yLabel):
    '''
    :param shortestWeightedPaths: shortest paths calculated based on an edge attribute, e.g. average distance
    :param description: e.g. 'pathTime', 'pathDist'
    :return:
    '''
    pathList = []
    for nodei in shortestWeightedPaths:
        nodejs = shortestWeightedPaths[nodei].keys()
        for nodej in nodejs:
            pathList.append([nodei, nodej, shortestWeightedPaths[nodei][nodej]['len']])
    dfPaths = pd.DataFrame(pathList, columns=['nodei', 'nodej', 'len'])

    binWidth = utils.calcBinWidth(dfPaths['len'].to_list())
    nBins = int((dfPaths['len'].max()-dfPaths['len'].min())/binWidth) + 1
    probs = dfPaths['len'].value_counts(normalize=True, bins=nBins)
    probs = probs.sort_index()
    # fits against asymmetric unimodal function
    binAvg = [(probs.index[i].left + probs.index[i].right)/2 for i in range(len(probs.index))]
    xObs = binAvg
    yObs = probs.values
    result = utils.fitAsymetricUnimodal(xObs, yObs, initA=1, initB=1, initC=1)

    summaryFilename = '%s/%s_modelFitSummary.txt' % (const.pathLenDistribFolder, description)
    plotFilename = '%s/%s_plot.png' % (const.pathLenDistribFolder, description)
    # saves result summary to text file
    with open(summaryFilename, 'w') as fh:
        fh.write(result.fit_report())

    # plots path attribs vs probability
    plt.plot(xObs, yObs, '.', color='black', label='raw data')
    plt.bar(xObs, yObs, width=int(binWidth*.95)) # only display 95% of binwidth in bar plot to better separate bins
    barLabels = [[int(probs.index[i].left+.5), int(probs.index[i].right+.5)] for i in range(len(probs.index))]
    plt.xticks(xObs, labels=barLabels, rotation='vertical')
    plt.plot(xObs, result.best_fit, '-', color='red', label='best fist')
    plt.xlabel(xLabel, fontsize=11)
    plt.ylabel(yLabel, fontsize=11)
    plt.tight_layout()
    #plt.show()
    plt.savefig(plotFilename, bbox_inches='tight', pad_inches=0)
    plt.clf()
