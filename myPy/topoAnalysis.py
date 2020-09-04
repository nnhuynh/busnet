import pickle
import time
import os
import pandas as pd

import constants as const
import topoNode
import topoPath
import utilities as utils

# ======================================================================================================================
def topoMain(G, dfRoutes):
    '''
    :param G:
    :return:
    '''

    # FIT NODE DEGREE DISTRIBUTION BY POWER LAW uses weighted and unweighted node degree

    # Only number of lines and number of services used as edge weight for this purpose
    # Using avg travel time and travel distance makes as weights in calculating node degree makes little sense.
    # NOTES AFTER FITTING:
    # - fitting of nlines~k and of nServices~k DON'T look good so no need to report them.
    # - fitting of p(k)~k using the power law looks VERY GOOD
    #nodeDeg = [[ node, G.nodes[node][const.GNodeAttribs.totDeg.name] ] for node in G.nodes]
    #topoNode.fitDegDistribPower(nodeDeg, 'degPower')
    #node_nLines = [[ node, G.nodes[node][const.GNodeAttribs.nLines.name] ] for node in G.nodes]
    #topoNode.fitDegDistribPower(node_nLines, 'nLinesPower')
    #node_nServices = [[ node, G.nodes[node][const.GNodeAttribs.nServices.name] ] for node in G.nodes]
    #topoNode.fitDegDistribPower(node_nServices, 'nServicesPower')

    # STEP 1. CALCULATES SHORTEST PATHS - RUN ON BUDF
    shortHops, shortDist, shortTime = topoPath.calcShortestPaths_v3(G)
    dfShortHops, dfShortDist, dfShortTime = topoPath.convertToDataframe(shortHops, shortDist, shortTime)
    with open('%s/dfShortHops.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfShortHops, f)
    with open('%s/dfShortDist.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfShortDist, f)
    with open('%s/dfShortTime.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfShortTime, f)

    # STEP 2a (CURRENTLY NOT IN USE). CALCULATES CLOSENESS CENTRALITY BY SOURCE AND BY TARGET - RUN ON BUDF
    '''
    # NOTE: closeness centrality by target and by source are not very different from each other.
    # Thus we only need to report on cc by target
    # calculating closeness centrality by source - Run on BUDF
    dfSourceCc = topoNode.calcSourceClosenessCentrality(G, shortHops, shortDist, shortTime)
    # calculating closeness centrality by target - Run on BUDF
    dfTargetCc = topoNode.calcTargetClosenessCentrality(G, shortHops, shortDist, shortTime)
    # dfTargetCc.to_csv('%s/dfTargetCc.csv' % const.outputsFolder, index=False)
    # dfSourceCc.to_csv('%s/dfSourceCc.csv' % const.outputsFolder, index=False)
    '''

    # STEP 2b. CALCULATES CENTRALITIES USING BUILT-IN NETWORKX FUNCTIONS - RUN ON BUDF
    '''
    NOTE: dfccTarget appeared to be very similar to dfTargetCc.
    # calculates betweenness centrality using built-in networkx functions - Run on BUDF
    dfcb = topoNode.calcBetweennessCentrality(G)
    # calculates closeness centrality (by target) using built-in networkx functions - Run on BUDF
    dfccTarget = topoNode.calcTargetClosenessCentrality_v2(G)
    # dfcb.to_csv('%s/dfcb.csv' % const.outputsFolder, index=False)
    # dfccTarget.to_csv('%s/dfccTarget.csv' % const.outputsFolder, index=False)
    '''

    # STEP 3. CALCULATES PROXIMITY DENSITY OF EACH NODE - RUN ON BUDF
    dfu30, dfu60, dfu90, dfo90, df3060, df6090 = topoNode.calcProxDensity(G, dfShortTime)
    with open('%s/dfu30.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfu30, f)
    with open('%s/dfu60.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfu60, f)
    with open('%s/dfu90.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfu90, f)
    with open('%s/dfo90.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(dfo90, f)
    with open('%s/df3060.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(df3060, f)
    with open('%s/df6090.pkl' % const.picklesFolder, 'wb') as f:
        pickle.dump(df6090, f)

    # assigns proximity density to each node in G.nodes
    assignProxDensityToG(G, dfu30, dfu60, dfu90, dfo90, df3060, df6090)

    # STEP 3b. CALCULATES BETWEENNESS CENTRALITY OF EACH NODE - RUN ON BUDF
    nProcesses = 4
    topoNode.calcBetwCentralMultip(G, nProcesses, dfShortTime, dfShortDist, dfShortHops)
    # assigns betweenness centrality to each node in G.nodes
    assignBetwCentralityToG(G)

    # STEP 4. ASSIGNS CALCULATED CENTRALITIES TO NODE ATTRIBUTES
    '''
    dfccTarget = pd.read_csv('%s/dfccTarget.csv' % const.outputsFolder,
                             usecols=['node', 'ccHopsNorm', 'ccDistNorm', 'ccTimeNorm'])
    dfcb = pd.read_csv('%s/dfcb.csv' % const.outputsFolder,
                       usecols=['node', 'cbHopsNorm', 'cbDistNorm', 'cbTimeNorm'])

    dfccTarget = dfccTarget.set_index('node')
    dfcb = dfcb.set_index('node')

    for node in G.nodes:
        G.nodes[node][const.GNodeAttribs.ccHops.name] = dfccTarget.at[node, 'ccHopsNorm']
        G.nodes[node][const.GNodeAttribs.ccDist.name] = dfccTarget.at[node, 'ccDistNorm']
        G.nodes[node][const.GNodeAttribs.ccTime.name] = dfccTarget.at[node, 'ccTimeNorm']
        G.nodes[node][const.GNodeAttribs.cbHops.name] = dfcb.at[node, 'cbHopsNorm']
        G.nodes[node][const.GNodeAttribs.cbDist.name] = dfcb.at[node, 'cbDistNorm']
        G.nodes[node][const.GNodeAttribs.cbTime.name] = dfcb.at[node, 'cbTimeNorm']
    '''

# ======================================================================================================================
def assignBetwCentralityToG(G):
    # consolidates betweenness centralities from multiprocessing
    dfbc = pd.DataFrame()
    nProcesses = 4
    for i in range(nProcesses):
        dfbcPart = pd.read_csv('%s/nodeCentrals/bc_%d.csv' % (const.outputsFolder, i))
        dfbc = pd.concat([dfbc, dfbcPart])

    dfbc.to_csv('%s/nodeCentrals/bc.csv' % const.outputsFolder)
    ndAttribs = [const.GNodeAttribs.cbTime.name, const.GNodeAttribs.cbDist.name, const.GNodeAttribs.cbHops.name]
    for node in G.nodes:
        for ndAttrib in ndAttribs:
            G.nodes[node][ndAttrib] = dfbc[ndAttrib].loc[dfbc['nodev']==node].values[0]

# ======================================================================================================================
def assignProxDensityToG(G, dfu30, dfu60, dfu90, dfo90, df3060, df6090):
    # columns in dfu30 are ['nodev', 'frV_u30', 'toV_u30', 'ndfrV_u30', 'ndtoV_u30']
    for node in G.nodes:
        # proximity density attribs for u30
        ndValAttribs = [const.GNodeAttribs.frV_u30.name, const.GNodeAttribs.toV_u30.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_u30.name, const.GNodeAttribs.ndtoV_u30.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = dfu30[ndAttrib].loc[dfu30['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = dfu30[ndAttrib].loc[dfu30['nodev'] == node].values[0].values

        # proximity density attribs for u60
        ndValAttribs = [const.GNodeAttribs.frV_u60.name, const.GNodeAttribs.toV_u60.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_u60.name, const.GNodeAttribs.ndtoV_u60.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = dfu60[ndAttrib].loc[dfu60['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = dfu60[ndAttrib].loc[dfu60['nodev'] == node].values[0].values

        # proximity density attribs for u90
        ndValAttribs = [const.GNodeAttribs.frV_u90.name, const.GNodeAttribs.toV_u90.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_u90.name, const.GNodeAttribs.ndtoV_u90.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = dfu90[ndAttrib].loc[dfu90['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = dfu90[ndAttrib].loc[dfu90['nodev'] == node].values[0].values

        # proximity density attribs for o90
        ndValAttribs = [const.GNodeAttribs.frV_o90.name, const.GNodeAttribs.toV_o90.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_o90.name, const.GNodeAttribs.ndtoV_o90.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = dfo90[ndAttrib].loc[dfo90['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = dfo90[ndAttrib].loc[dfo90['nodev'] == node].values[0].values

        # proximity density attribs for 3060
        ndValAttribs = [const.GNodeAttribs.frV_3060.name, const.GNodeAttribs.toV_3060.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_3060.name, const.GNodeAttribs.ndtoV_3060.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = df3060[ndAttrib].loc[df3060['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = df3060[ndAttrib].loc[df3060['nodev'] == node].values[0].values

        # proximity density attribs for 6090
        ndValAttribs = [const.GNodeAttribs.frV_6090.name, const.GNodeAttribs.toV_6090.name]
        ndListAttribs = [const.GNodeAttribs.ndfrV_6090.name, const.GNodeAttribs.ndtoV_6090.name]
        for ndAttrib in ndValAttribs:
            G.nodes[node][ndAttrib] = df6090[ndAttrib].loc[df6090['nodev'] == node].values[0]
        for ndAttrib in ndListAttribs:
            G.nodes[node][ndAttrib] = df6090[ndAttrib].loc[df6090['nodev'] == node].values[0].values

# ======================================================================================================================
# ======================================================================================================================
def draft(G):
    '''
    '''
    pass
    '''
    # FIT NODE DEGREE DISTRIBUTION BY EXPONENTIAL LAW uses weighted and unweighted node degree
    # Only number of lines and number of services used as edge weight for this purpose
    # Using avg travel time and travel distance makes as weights in calculating node degree makes little sense.
    topoNode.fitDegDistribExponential(totDeg, 'degExp')
    topoNode.fitDegDistribExponential(totDeg_line, 'nLinesDegExp')
    topoNode.fitDegDistribExponential(totDeg_serv, 'nServicesDegExp')
    '''

    '''
    # Todo: FIT NODE STRENGTH VS NODE DEGREE USING POWER LAW
    print('start fitNdStrengthDegPower')
    topoNode.fitNdStrengthDegPower(totDeg_line, totDeg, 'k', 's_nlines(k)', '%s/nlines_vs_k.png'%const.nodeStrDegFolder)
    print('end fitNdStrengthDegPower')
    #fitNdStrengthDegPower(totDeg_serv, totDeg)
    '''

    '''
    # Todo: FIT NODE STRENGTH VS NODE DEGREE USING EXPONENTIAL LAW
    #fitNdStrengthDegExp(totDeg_line, totDeg)
    #fitNdStrengthDegExp(totDeg_serv, totDeg)
    '''

    '''
    # CALCULATES PATH LENGTH - WARNING: This will take a VERY long time!!!
    # unweighted and using time and distance as weight
    # - calculates shortest paths of all node pairs
    # - calculates and plots the distribution of shortest distances
    # - calculates average of shortest paths, aka shortest path length.
    # IMPORTANT: formula used in calculating shortest path length may be different for unweighted and weighted edges.
    # Also note that our graph is directed.
    # - calculates diameter of the network, aka longest of the shortest paths

    shortestPaths = topoPath.calcShortestPaths(G, nodeList, edgeWeight=None)
    shortestPaths_time = topoPath.calcShortestPaths(G, nodeList, edgeWeight=const.GEdgeAttribs.meanTravTime.name)
    shortestPaths_dist = topoPath.calcShortestPaths(G, nodeList, edgeWeight=const.GEdgeAttribs.meanRoadDist.name)
    #print(shortestPaths['BX14']['BX_06']['path'])
    #print(shortestPaths['BX14']['BX_06']['len'])

    # dumps shortest paths to pickle files
    utils.picklePathLens(shortestPaths, shortestPaths_time, shortestPaths_dist)

    topoPath.calcPathLenDistrib(shortestPaths, 'pathLen', 'Path length l', 'p(l)')
    topoPath.calcPathDistrib(shortestPaths_time, 'pathAvgTime', 'tAvg (s)', 'p(tAvg)')
    topoPath.calcPathDistrib(shortestPaths_dist, 'pathAvgDist', 'dAvg (m)', 'p(dAvg)')
    '''