import pickle
import os
import pandas as pd
import networkx as nx

import graph_v2
import constants as const
import topologicalAnalysis as topo
import geoPlotter

import ttGraph
import ttGraphVisual
import ttGraphAnalysis

# ======================================================================================================================
if __name__=='__main__':

    # CONSTRUCTS GRAPH IN L-SPACE AND DFROUTES =========================================================================
    pFile_G = '%s/G.pkl' % const.picklesFolder
    pFile_dfRoutes = '%s/dfRoutes.pkl' % const.picklesFolder
    if os.path.isfile(pFile_G) and os.path.isfile(pFile_dfRoutes):
        G = pickle.load(open(pFile_G, 'rb'))
        dfRoutes = pickle.load(open(pFile_dfRoutes, 'rb'))
        print('unpickling G and dfRoutes completed')
    else:
        G, dfRoutes = graph_v2.construct_G_dfRoutes()
        print('constructing G and dfRoutes completed')
        with open(pFile_G, 'wb') as f:
            pickle.dump(G, f)
        with open(pFile_dfRoutes, 'wb') as f:
            pickle.dump(dfRoutes, f)
    #printStuff(G,dfRoutes)


    # TOPOLOGICAL ANALYSIS =============================================================================================
    #topo.topoMain(G, dfRoutes)

    # WEBMAP PLOTTING
    # plot all directed routes on map, allowing for selection and displaying of individual routes
    #geoPlotter.plotRoutesWithLayers(G, dfRoutes, geoPlotter.makeBaseMap(), '%s/routesAllLayers.html' % const.mapPlots)

    # colour coded map of degree, nLines, and nServices of all nodes
    #geoPlotter.plotNodeStrengths(G, geoPlotter.makeBaseMap(), '%s/nodeStrengths.html' % const.mapPlots)

    # colour coded map of closeness centralities of all nodes
    #geoPlotter.plotClosenessCentralities(G, geoPlotter.makeBaseMap(), '%s/nodeClosenessCentrals.html' % const.mapPlots)

    # colour coded map of betweenness centralities of all nodes
    #geoPlotter.plotBetweennessCentralities(G, geoPlotter.makeBaseMap(), '%s/nodeBetweennessCentrals.html' % const.mapPlots)

    # colour coded map of proximity density of all nodes
    #geoPlotter.plotProximityDensity(G, geoPlotter.makeBaseMap(), '%s/proxDensities.html' % const.mapPlots)


    # TOPOLOGICAL TEMPORAL ANALYSIS ====================================================================================
    Gtt, dfNdDetails, ndIDict = ttGraph.makeTopoTempGraph_v2(dfRoutes)
    startTime = 5*3600 # 5am from midnight in seconds
    tWidth = 3*3600 # 3 hours in seconds
    ndPairsTried = []
    GttSub, shortestTimes, ndPairsTried = ttGraphAnalysis.calcShortestPaths(
        Gtt, dfNdDetails, ndIDict, [startTime, startTime + tWidth], G, ndPairsTried)

    with open('%s/GttSub_5am_8am.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(GttSub, f)
    with open('%s/shortTimes_5am_8am.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(shortestTimes, f)
    with open('%s/ndPairsTried_to_8am.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(ndPairsTried, f)

    print('yay')