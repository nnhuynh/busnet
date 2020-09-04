import pickle
import os
import pandas as pd
import networkx as nx
import numpy as np
from geopy.distance import distance
import time
import matplotlib
import shutil

import graph_v2
import constants as const
import topoAnalysis as topo
import topoNode
import topoPostproc
import geoPlotter
import utilities as utils

import ttGraph_v2
import ttgAnalysisMultip_v2
import ttgVisual_v2
import ttgPostprocMultip

# ======================================================================================================================
if __name__=='__main__':
    '''
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
    '''

    '''
    stopPairsSameRoute = ttgAnalysisMultip.getStopPairsSameRoute(dfRoutes)
    with open('%s/stopPairsSameRoute.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(stopPairsSameRoute, f)
    '''

    '''
    # TOPOLOGICAL ANALYSIS =============================================================================================
    #topo.topoMain(G, dfRoutes)
    # assigns betweenness centralities (by time, by distance, by number of hops) to G
    G = pickle.load(open('%s/G_proxDens.pkl' % const.picklesFolder, 'rb'))
    topo.assignBetwCentralityToG(G)
    with open('%s/G_wNdAttribs.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(G, f)
    '''

    '''
    # TOPO ANLYSIS POSTPROCESSING
    G_wNdAttribs = pickle.load(open('%s/G_wNdAttribs.pkl' % const.picklesFolder, 'rb'))
    dfRoutes = pickle.load(open( '%s/dfRoutes.pkl' % const.picklesFolder, 'rb'))
    stopPairDist = pickle.load(open('%s/stopPairsDist.pkl' % const.picklesFolder, 'rb'))
    topoPostproc.main_topoPostproc(G_wNdAttribs, dfRoutes, stopPairDist)
    '''

    '''
    # TEMPORAL TOPOLOGICAL ANALYSIS ====================================================================================
    #dirRouteID = '10_1'
    #timeWindow = [5*3600, 22*3600]
    #fileName = '%s/%s_allDay.png' % (const.trashSite, dirRouteID)
    #ttgVisual_v2.plotTransitLinks_v2(gttTransit, dirRouteID, timeWindow, fileName)

    #dfRoutes = pickle.load(open('%s/dfRoutes.pkl' % const.picklesFolder, 'rb'))
    #ttgVisual_v2.plotRoutesAllday(dfRoutes)

    #ttgPostprocMultip.calcBetwCentral(allStops=None, dfAllShortestPaths=None)
    '''

    ttgPostprocMultip.main_ttgPostprocMultip()
    #ttgPostprocMultip.main_draft()

    '''
    df = pd.DataFrame({'a': [1.2, 2.3, 4.5],
                       'b': [2.1, 0.2, 5.3],})

    dfTmp = pd.DataFrame()
    dfTmp['tly'] = abs(df['a'] - df['b'])
    print(dfTmp)
    '''

    print('yay')