import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import time
import pickle

import geoPlotter
import constants as const
import utilities as utils
import topoNode

# ======================================================================================================================
def createMapPlots(G, dfRoutes, dfStopAttribs):
    # plot all directed routes on map, allowing for selection and displaying of individual routes
    #geoPlotter.plotRoutesWithLayers(G, dfRoutes, geoPlotter.makeBaseMap(), '%s/routesAllLayers.html' % const.mapPlots)
    # colour coded map of degree, nLines, and nServices of all nodes
    #geoPlotter.plotNodeStrengths(G, geoPlotter.makeBaseMap(), '%s/nodeStrengths.html' % const.mapPlots)
    # colour coded map of closeness centralities of all nodes
    #geoPlotter.plotClosenessCentralities(G, geoPlotter.makeBaseMap(), '%s/nodeClosenessCentrals.html' % const.mapPlots)
    # colour coded map of betweenness centralities of all nodes
    #geoPlotter.plotBetweennessCentralities(G, dfStopAttribs, const.GNodeAttribs.cbTime,
    #                                       geoPlotter.makeBaseMap(), '%s/betwCentralTime.html' % const.mapPlots)
    # colour coded map of proximity density of all nodes
    #geoPlotter.plotProximityDensity(G, dfStopAttribs, geoPlotter.makeBaseMap(), '%s/proxDensities.html'%const.mapPlots)

    # plots group of stops by distance from the network centroid
    latCentroid, lngCentroid = utils.calcCentroidOfStops(G)
    geoPlotter.plotStopsByRange(G, dfRoutes, dfStopAttribs, latCentroid, lngCentroid, geoPlotter.makeBaseMap(),
                                '%s/stopsByDistFromCentroid.html' % const.mapPlots)


# ======================================================================================================================
def mkBoxplotNdAttribs(dfNodeAttribs, plotAttribs, xAttrib):
    for attrib in plotAttribs:
        dfNodeAttribs.boxplot(column=[attrib], by=[xAttrib])
        plt.savefig('%s/plots/%s~%s.png' % (const.nodeCentralsFolder, attrib, xAttrib), bbox_inches='tight', pad_inches=1)

    '''
    plotAttribs = ['cbHops', 'cbDist', 'cbTime', 'toV_u30', 'toV_3060', 'toV_6090', 'toV_u60', 'toV_u90', 'toV_o90']
    for attrib in plotAttribs:
        fig, ax = plt.subplots()
        ax.scatter(dfNodeAttribs['nLines'], dfNodeAttribs[attrib], marker='o', s=3)
        ax.set_ylim([0, 1])
        plt.savefig('%s/plots/%s~nLines.png' % (const.nodeCentralsFolder, attrib), bbox_inches='tight', pad_inches=1)
    '''

# ======================================================================================================================
def mkViolinPlotsProxDens(dfNdAttribs, rangeCats, ndAttribs, yAxisTitle, filename):
    dfSns = pd.DataFrame()

    dfTmp = pd.DataFrame()
    dfTmp = dfNdAttribs[[ndAttribs[0],'range']]
    dfTmp[ndAttribs[0]] = dfTmp[ndAttribs[0]] * (dfNdAttribs.shape[0] - 1)
    dfTmp[ndAttribs[0]].astype(int)
    dfTmp = dfTmp.assign(type = 'asOrigin')
    dfTmp = dfTmp.rename(columns={ndAttribs[0]: yAxisTitle})

    dfSns = pd.concat([dfSns, dfTmp])

    dfTmp = pd.DataFrame()
    dfTmp = dfNdAttribs[[ndAttribs[1], 'range']]
    dfTmp[ndAttribs[1]] = dfTmp[ndAttribs[1]] * (dfNdAttribs.shape[0] - 1)
    dfTmp[ndAttribs[1]].astype(int)
    dfTmp = dfTmp.assign(type='asDestination')
    dfTmp = dfTmp.rename(columns={ndAttribs[1]: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    fig = plt.figure(figsize=(5,5))

    sns.set(style="whitegrid")
    ax = sns.violinplot(x="range", y=yAxisTitle, hue='type', data=dfSns, palette='muted', order=rangeCats,
                        split=True, inner='quartile', cut=0, scale='count')
    #ax.set_xticklabels(rangeCats, rotation=90)
    ax.set_ylim([0, dfNdAttribs.shape[0]])
    ax.yaxis.grid(True)

    ax.set_xlabel('Distance to the bus network centroid')

    fig.savefig(filename, bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def mkBoxplotsProxDens(dfNdAttribs, rangeCats, ndAttribs, yAxisTitle, filename):
    dfSns = pd.DataFrame()

    dfTmp = pd.DataFrame()
    dfTmp = dfNdAttribs[[ndAttribs[0],'range']]
    dfTmp[ndAttribs[0]] = dfTmp[ndAttribs[0]] * (dfNdAttribs.shape[0] - 1)
    dfTmp[ndAttribs[0]].astype(int)
    dfTmp = dfTmp.assign(type = 'asOrigin')
    dfTmp = dfTmp.rename(columns={ndAttribs[0]: yAxisTitle})

    dfSns = pd.concat([dfSns, dfTmp])

    dfTmp = pd.DataFrame()
    dfTmp = dfNdAttribs[[ndAttribs[1], 'range']]
    dfTmp[ndAttribs[1]] = dfTmp[ndAttribs[1]] * (dfNdAttribs.shape[0] - 1)
    dfTmp[ndAttribs[1]].astype(int)
    dfTmp = dfTmp.assign(type='asDestination')
    dfTmp = dfTmp.rename(columns={ndAttribs[1]: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    fig = plt.figure(figsize=(5,5))

    sns.set(style="whitegrid")
    ax = sns.boxplot(x="range", y=yAxisTitle, hue='type', data=dfSns, palette='muted', order=rangeCats, fliersize=.5)
    #ax.set_xticklabels(rangeCats, rotation=90)
    ax.set_ylim([0, dfNdAttribs.shape[0]])
    ax.yaxis.grid(True)

    ax.set_xlabel('Distance to the bus network centroid')

    fig.savefig(filename, bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def plotHist(dfNdAttribs, plotAttribs):
    for attrib in plotAttribs:
        fig, ax = plt.subplots()
        ax.hist(dfNdAttribs[attrib])
        plt.savefig('%s/plots/hist/%s.png' % (const.nodeCentralsFolder, attrib), bbox_inches='tight', pad_inches=1)


# ======================================================================================================================
def mkBoxplotProxDensVsR(df, rangeCats, plotAttribs):
    groups = df.groupby(['range'])

    for attrib in plotAttribs:
        fig, ax = plt.subplots()
        for position,cat in enumerate(rangeCats):
            for name, group in groups:
                if name==cat:
                    ax.boxplot(group[attrib], positions=[position])
                    break
        ax.set_xticks(range(len(rangeCats) + 1))
        ax.set_xticklabels(rangeCats, rotation=90)
        ax.set_ylim([0,1])
        ax.yaxis.grid(True)
        plt.savefig('%s/plots/%s.png' % (const.nodeCentralsFolder, attrib), bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def mkDfNodeAttribs(G):
    latCentroid, lngCentroid = utils.calcCentroidOfStops(G)

    ndAttribsVsR = []
    for node in G.nodes:
        lng = G.nodes[node][const.GNodeAttribs.Lng.name]
        lat = G.nodes[node][const.GNodeAttribs.Lat.name]
        ndAttribsVsR.append([node,
                             utils.calcGeodesicDist([lng, lat], [lngCentroid, latCentroid]) / 1000,  # in km
                             lat,
                             lng,

                             G.nodes[node][const.GNodeAttribs.totDeg.name],
                             G.nodes[node][const.GNodeAttribs.nLines.name],
                             G.nodes[node][const.GNodeAttribs.nServices.name],

                             G.nodes[node][const.GNodeAttribs.cbHops.name],
                             G.nodes[node][const.GNodeAttribs.cbDist.name],
                             G.nodes[node][const.GNodeAttribs.cbTime.name],

                             G.nodes[node][const.GNodeAttribs.toV_u30.name],
                             G.nodes[node][const.GNodeAttribs.frV_u30.name],
                             G.nodes[node][const.GNodeAttribs.toV_3060.name],
                             G.nodes[node][const.GNodeAttribs.frV_3060.name],
                             G.nodes[node][const.GNodeAttribs.toV_6090.name],
                             G.nodes[node][const.GNodeAttribs.frV_6090.name],
                             G.nodes[node][const.GNodeAttribs.toV_o90.name],
                             G.nodes[node][const.GNodeAttribs.frV_o90.name],
                             G.nodes[node][const.GNodeAttribs.toV_u60.name],
                             G.nodes[node][const.GNodeAttribs.frV_u60.name],
                             G.nodes[node][const.GNodeAttribs.toV_u90.name],
                             G.nodes[node][const.GNodeAttribs.frV_u90.name],

                             G.nodes[node][const.GNodeAttribs.ndtoV_u30.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_u30.name],
                             G.nodes[node][const.GNodeAttribs.ndtoV_3060.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_3060.name],
                             G.nodes[node][const.GNodeAttribs.ndtoV_6090.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_6090.name],
                             G.nodes[node][const.GNodeAttribs.ndtoV_o90.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_o90.name],
                             G.nodes[node][const.GNodeAttribs.ndtoV_u60.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_u60.name],
                             G.nodes[node][const.GNodeAttribs.ndtoV_u90.name],
                             G.nodes[node][const.GNodeAttribs.ndfrV_u90.name]
                             ])
    df = pd.DataFrame(ndAttribsVsR, columns=['StationId', 'R', 'Lat', 'Lng',
                                             'totDeg', 'nLines', 'nServices',
                                             'cbHops', 'cbDist', 'cbTime',

                                             'toV_u30', 'frV_u30',
                                             'toV_3060', 'frV_3060',
                                             'toV_6090', 'frV_6090',
                                             'toV_o90', 'frV_o90',
                                             'toV_u60', 'frV_u60',
                                             'toV_u90', 'frV_u90',

                                             'ndtoV_u30', 'ndfrV_u30',
                                             'ndtoV_3060', 'ndfrV_3060',
                                             'ndtoV_6090', 'ndfrV_6090',
                                             'ndtoV_o90', 'ndfrV_o90',
                                             'ndtoV_u60', 'ndfrV_u60',
                                             'ndtoV_u90', 'ndfrV_u90'])
    # df.to_csv('%s/nodeCentrals/proxDensVsR.csv' % const.outputsFolder, index=False)
    df['range'] = df['R'].apply(utils.getRange)
    return df

# ======================================================================================================================
def mkDfTmp(stop1, stop2List):
    stop1List = [stop1] * len(stop2List)
    dfTmp = pd.DataFrame({'stop1': stop1List, 'stop2': stop2List})
    return dfTmp

def calcDistanceDistrib(dfNdAttribs, stopPairDist):
    print('calcDistanceDistrib started')
    proxDensCats = ['ndfrV_u30', 'ndfrV_3060', 'ndfrV_6090', 'ndfrV_u60', 'ndfrV_u90', 'ndfrV_o90',
                    'ndtoV_u30', 'ndtoV_3060', 'ndtoV_6090', 'ndtoV_u60', 'ndtoV_u90', 'ndtoV_o90']
    starttime = time.perf_counter()
    for proxDensCat in proxDensCats:
        dfStopPairDist = pd.DataFrame()
        for idx, row in dfNdAttribs.iterrows():
            stop1 = row['StationId']
            stop2List = row[proxDensCat].tolist()
            dfStopPairDist = pd.concat([dfStopPairDist, mkDfTmp(stop1,stop2List)])
        dfStopPairDist['stopPair'] = dfStopPairDist.apply(lambda row: '%d_%d' % (row['stop1'],row['stop2']), axis=1)
        dfStopPairDist['stopPairDist'] = dfStopPairDist['stopPair'].map(stopPairDist)
        dfStopPairDist['stopPairDist'] = dfStopPairDist['stopPairDist'] * 1e-3 # converts to km
        stop1Groups = dfStopPairDist.groupby(['stop1'])['stopPairDist'].mean()
        dfNdAttribs = pd.merge(dfNdAttribs, stop1Groups.to_frame(), how='left', left_on=['StationId'], right_on=['stop1'])
        dfNdAttribs = dfNdAttribs.rename(columns = {'stopPairDist': 'avgDist_%s' % proxDensCat})
        print('\t%s completed - took %.4g mins' % (proxDensCat, (time.perf_counter() - starttime)/60))

    return dfNdAttribs

# ======================================================================================================================
def plotDistanceDistrib(dfNdAttribs, rangeCats, distFrAttrib, distToAttrib, yAxisTitle, filename):
    dfSns = pd.DataFrame()

    dfTmp = dfNdAttribs[[distFrAttrib, 'range']]
    dfTmp = dfTmp.assign(type='asOrigin')
    dfTmp = dfTmp.rename(columns={distFrAttrib: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    dfTmp = dfNdAttribs[[distToAttrib, 'range']]
    dfTmp = dfTmp.assign(type='asDestination')
    dfTmp = dfTmp.rename(columns={distToAttrib: yAxisTitle})
    dfSns = pd.concat([dfSns, dfTmp])

    fig = plt.figure()

    sns.set(style="whitegrid")
    ax = sns.violinplot(x="range", y=yAxisTitle, hue='type', data=dfSns,
                        palette='muted', order=rangeCats, split=True, inner='quartile', cut=0, scale='count')
    # ax.set_xticklabels(rangeCats, rotation=90)
    #ax.set_ylim([0, 1])
    ax.yaxis.grid(True)
    fig.savefig(filename, bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def plotBoxplotsVsNdDegree(dfStopStrengths, ndAttrib, ytitle, ax):
    #fig = plt.figure(figsize=(10, 5))
    sns.boxplot(x='totDeg', y=ndAttrib, data=dfStopStrengths, color=mcolors.CSS4_COLORS['lightgrey'],
                fliersize=1, ax=ax)
    ax.set_ylabel(ytitle)
    ax.set_xlabel('Node degree, k')
    ax.yaxis.grid(True)
    #fig.savefig('%s/nLines~k.png' % const.nodeDegDistribFolder, bbox_inches='tight', pad_inches=1)

# ======================================================================================================================
def main_topoPostproc(G_wNdAttribs, dfRoutes, stopPairDist):
    '''
    :param G_wNdAttribs:
    :param dfRoutes:
    :return:
    '''
    dfStopAttribs = mkDfNodeAttribs(G_wNdAttribs)
    #rangeCats = ['under 5km', '5km - 10km', '10km - 15km', '15km - 20km', '20km - 30km', '30km - 40km', 'over 40km']
    rangeCats = ['under 10km', '10km - 20km', 'over 20km']

    # CREATES MAPPLOTS
    createMapPlots(G_wNdAttribs, dfRoutes, dfStopAttribs)

    '''
    # CREATES NODE DEGREE PLOT AND BOX PLOT OF NUMBER OF LINES AND NUMBER OF SERVICES AT A NODE VERSUS NODE DEGREE
    fig, ax = plt.subplots(1, 3, figsize=(10,5), constrained_layout=True)
    
    # node degree
    # nodeDeg = dfStopAttribs[['StationId', 'totDeg']].values.tolist()
    nodeDeg = dfStopAttribs[['StationId', 'totDeg']].loc[dfStopAttribs['totDeg']>1].values.tolist()
    topoNode.fitDegDistribPower(nodeDeg, 'degPower', axes = ax[0])

    # boxplots
    dfStopStrengths = dfStopAttribs[['StationId', 'totDeg', 'nLines', 'nServices']].loc[dfStopAttribs['totDeg'] > 1]
    plotBoxplotsVsNdDegree(dfStopStrengths, ndAttrib='nLines', ytitle='Number of lines', ax=ax[1])
    plotBoxplotsVsNdDegree(dfStopStrengths, ndAttrib='nServices', ytitle='Number of services', ax=ax[2])
    fig.savefig('%s/nodeDeg.png' % const.nodeDegDistribFolder)
    '''

    '''
    mkBoxplotNdAttribs(dfStopAttribs, plotAttribs = ['cbHops', 'cbDist', 'cbTime', 'nLines', 'nServices'], 
                       xAttrib = 'totDeg')
    plotHist(dfStopAttribs, plotAttribs = ['nLines', 'nServices', 'cbDist', 'cbHops', 'cbTime'])
    '''

    '''
    # PLOTS OF PROXIMITY DENSITY
    #mkBoxplotProxDensVsR(G_wNdAttribs, rangeCats,
    #                     plotAttribs = ['toV_u30', 'toV_3060', 'toV_6090', 'toV_o90', 'toV_u60', 'toV_u90'])
    boxplotFolder = '%s/plots/boxplotsProxDens' % const.nodeCentralsFolder
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_u30', 'toV_u30'],
                       yAxisTitle='Proximity density - under 30 minutes',
                       filename='%s/proxDens_u30.png' % boxplotFolder)
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_3060', 'toV_3060'],
                       yAxisTitle='Proximity density - 30-60 minutes',
                       filename='%s/proxDens_3060.png' % boxplotFolder)
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_6090', 'toV_6090'],
                       yAxisTitle='Proximity density - 60-90 minutes',
                       filename='%s/proxDens_6090.png' % boxplotFolder)
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_u60', 'toV_u60'],
                       yAxisTitle='Proximity density - under 60 minutes',
                       filename='%s/proxDens_u60.png' % boxplotFolder)
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_u90', 'toV_u90'],
                       yAxisTitle='Proximity density - under 90 minutes',
                       filename='%s/proxDens_u90.png' % boxplotFolder)
    mkBoxplotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_o90', 'toV_o90'],
                       yAxisTitle='Proximity density - over 90 minutes',
                       filename='%s/proxDens_o90.png' % boxplotFolder)
    '''
    '''
    violinPlotFolder = '%s/plots/violinsProxDens' % const.nodeCentralsFolder
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs = ['frV_u30', 'toV_u30'],
                          yAxisTitle = 'Proximity density - under 30 minutes',
                          filename = '%s/proxDens_u30.png' % violinPlotFolder)
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_3060', 'toV_3060'],
                          yAxisTitle='Proximity density - 30-60 minutes',
                          filename='%s/proxDens_3060.png' % violinPlotFolder)
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_6090', 'toV_6090'],
                          yAxisTitle='Proximity density - 60-90 minutes',
                          filename='%s/proxDens_6090.png' % violinPlotFolder)
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_u60', 'toV_u60'],
                          yAxisTitle='Proximity density - under 60 minutes',
                          filename='%s/proxDens_u60.png' % violinPlotFolder)
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_u90', 'toV_u90'],
                          yAxisTitle='Proximity density - under 90 minutes',
                          filename='%s/proxDens_u90.png' % violinPlotFolder)
    mkViolinPlotsProxDens(dfStopAttribs, rangeCats, ndAttribs=['frV_o90', 'toV_o90'],
                          yAxisTitle='Proximity density - over 90 minutes',
                          filename='%s/proxDens_o90.png' % violinPlotFolder)
    '''

    '''
    # PLOTS OF DISTANCE DISTRIBUTION BASED ON TEMPORAL PROXIMITY 
    #calcDistanceDistrib(dfStopAttribs, stopPairDist)
    #plotDistanceDistrib(rangeCats)
    dfStopAttribs = calcDistanceDistrib(dfStopAttribs, stopPairDist)
    stopPairDistFolder = '%s/plots/stopPairDist' % const.nodeCentralsFolder
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_u30', 'avgDist_ndtoV_u30',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_u30.png' % stopPairDistFolder)
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_3060', 'avgDist_ndtoV_3060',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_3060.png' % stopPairDistFolder)
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_6090', 'avgDist_ndtoV_6090',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_6090.png' % stopPairDistFolder)
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_u60', 'avgDist_ndtoV_u60',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_u60.png' % stopPairDistFolder)
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_u90', 'avgDist_ndtoV_u90',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_u90.png' % stopPairDistFolder)
    plotDistanceDistrib(dfStopAttribs, rangeCats, 'avgDist_ndfrV_o90', 'avgDist_ndtoV_o90',
                           'Average distance between stop pairs (km)',
                           '%s/stopPairDist_o90.png' % stopPairDistFolder)
    '''