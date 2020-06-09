import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.features import GeoJson, GeoJsonTooltip
from folium.plugins import MarkerCluster
import branca.colormap as cm

import pandas as pd
import constants as const

# ======================================================================================================================
def makeBaseMap():
    latlon = [10.77679,106.705856]
    #m = folium.Map(location=latlon)
    m = folium.Map(location=latlon, tiles='cartodbpositron', zoom_start=11, prefer_canvas=True)
    #m = folium.Map(location=latlon, tiles='Stamen Terrain')
    #m = folium.Map(location=latlon, tiles='Stamen Toner')
    #m.save('hcmc.html')
    return m

# ======================================================================================================================
def addMarkers(G, m):
    for node in G.nodes():
        lat = G.nodes[node][const.GNodeAttribs.Lat.name]
        long = G.nodes[node][const.GNodeAttribs.Lng.name]
        address = G.nodes[node][const.GNodeAttribs.Address.name]
        folium.Marker(location=[lat,long],
                      popup = address,
                      icon=folium.Icon()).add_to(m)
    m.save('hcmc1.html')
    return m

# ======================================================================================================================
def plotRouteToBase(G, dfTimetable, myMap):
    for iidx in range(len(dfTimetable.index)):
        stnOrder = dfTimetable.index[iidx]
        crnStnCode = dfTimetable.at[stnOrder,const.RoutesTtbCols.StationCode.name]
        coordCrnStn = [G.nodes[crnStnCode][const.GNodeAttribs.Lat.name],
                       G.nodes[crnStnCode][const.GNodeAttribs.Lng.name]]
        # adds marker to bus stops
        folium.CircleMarker(location=coordCrnStn,
                            radius = 1, color='blue', weight = 2, opacity=.3, fill_color='blue', fill_opacity=.3,
                            tooltip = 'StationId %d, %s' % (G.nodes[crnStnCode][const.GNodeAttribs.StationId.name],
                                                            G.nodes[crnStnCode][const.GNodeAttribs.StationDesc.name])).add_to(myMap)
        if iidx==0:
            continue

        # gets pathPoints between crnStnCode and preStnCode
        prevStnOrder = dfTimetable.index[iidx-1]
        preStnCode = dfTimetable.at[prevStnOrder,const.RoutesTtbCols.StationCode.name]
        edgeRoutes = G.edges[preStnCode,crnStnCode][const.GEdgeAttribs.routes.name]
        pathPoints = edgeRoutes[const.GEdgeRouteCols.pathPoints.name][0]

        # notes that coordinates of each path point are in long,lat order, thus need to convert this to lat,long order
        points = [(point[1], point[0]) for point in pathPoints]

        # adds polyline of this route to map
        pline = folium.PolyLine(points, color='blue', weight=2, opacity=.3, popup='route info')
        pline.add_to(myMap)

# ======================================================================================================================
def plotRouteToOverLay(G, dfTimetable, routeDesc):
    routeLayer = FeatureGroup(name=routeDesc, show=False)
    for iidx in range(len(dfTimetable.index)):
        stnOrder = dfTimetable.index[iidx]
        crnStnCode = dfTimetable.at[stnOrder,const.RoutesTtbCols.StationCode.name]
        coordCrnStn = [G.nodes[crnStnCode][const.GNodeAttribs.Lat.name],
                       G.nodes[crnStnCode][const.GNodeAttribs.Lng.name]]
        # adds marker to bus stops
        folium.Marker(location=coordCrnStn,
                      tooltip='StationId %d, %s' % (G.nodes[crnStnCode][const.GNodeAttribs.StationId.name],
                                                    G.nodes[crnStnCode][const.GNodeAttribs.StationDesc.name]),
                      icon=folium.Icon(prefix='fa', icon='bus')).add_to(routeLayer)
        if iidx == 0:
            continue

        # gets pathPoints between crnStnCode and preStnCode
        prevStnOrder = dfTimetable.index[iidx-1]
        preStnCode = dfTimetable.at[prevStnOrder, const.RoutesTtbCols.StationCode.name]
        edgeRoutes = G.edges[preStnCode, crnStnCode][const.GEdgeAttribs.routes.name]
        pathPoints = edgeRoutes[const.GEdgeRouteCols.pathPoints.name][0]

        # notes that coordinates of each path point are in long,lat order, thus need to convert this to lat,long order
        points = [(point[1], point[0]) for point in pathPoints]

        # adds polyline of this route to map
        folium.PolyLine(points, color='red', weight=3, opacity=.3, popup='route info').add_to(routeLayer)

    return routeLayer

# ======================================================================================================================
def plotRoutesWithLayers(G, dfRoutes, myMap, mapfile):
    for index,route in dfRoutes.iterrows():
        designedTtb = route[const.RoutesCols.designedTtb.name]
        tupleList = list(zip(designedTtb[const.RoutesTtbCols.StationOrder.name],
                             designedTtb[const.RoutesTtbCols.StationCode.name]))
        dfDesignedTtb = pd.DataFrame(tupleList, columns=[const.RoutesTtbCols.StationOrder.name,
                                                         const.RoutesTtbCols.StationCode.name])
        dfDesignedTtb = dfDesignedTtb.set_index(const.RoutesTtbCols.StationOrder.name)
        dfDesignedTtb.sort_index()

        plotRouteToBase(G, dfDesignedTtb, myMap)

        routeDesc = 'route %s, direction %s' % (route[const.RoutesCols.RouteNo.name],
                                                route[const.RoutesCols.StationDirection.name])

        routeLayer = plotRouteToOverLay(G, dfDesignedTtb, routeDesc)
        routeLayer.add_to(myMap)

    folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)

    LayerControl().add_to(myMap)
    myMap.save(mapfile)

# ======================================================================================================================
def plotNodeFeature(G, dfNdAttribs, ndAttrib, myMap, colourmap=None):
    feature = FeatureGroup(name=ndAttrib.value, show=False)

    if colourmap==None:
        minVal = dfNdAttribs[ndAttrib.name].min()
        maxVal = dfNdAttribs[ndAttrib.name].max()
        colourmap = cm.linear.Reds_09.scale(minVal, maxVal)
        colourmap.caption = ndAttrib.value

    # sorts nodes by value so that nodes with larger values plot later
    # this ensures that circles with darker colour (larger value) always on top thus more visible.
    dfNdAttribs = dfNdAttribs.sort_values(by=[ndAttrib.name])

    for idx,row in dfNdAttribs.iterrows():
        node = row[const.GNodeAttribs.StationId.name]
        deg = row[ndAttrib.name]
        coordCrnStn = [row[const.GNodeAttribs.Lat.name], row[const.GNodeAttribs.Lng.name]]

        html = """<html> {ndAttrib} {val} </html><br>
                StationId {stnid}<br>
                {ndDesc}
                """.format(ndAttrib = ndAttrib.value, val = float("{:.3f}".format(deg)),
                           stnid = G.nodes[node][const.GNodeAttribs.StationId.name],
                           ndDesc = G.nodes[node][const.GNodeAttribs.StationDesc.name])

        folium.CircleMarker(location=coordCrnStn,
                            radius=5, color=colourmap(deg), weight=2, opacity=0.5, fill_color=colourmap(deg),
                            fill_opacity=.7,
                            tooltip=html).add_to(feature)

    myMap.add_child(colourmap)
    feature.add_to(myMap)
    return myMap

# ======================================================================================================================
def plotNodeStrengths(G, myMap, mapfile):
    nodeStrengths = [[node,
                      G.nodes[node][const.GNodeAttribs.Lat.name],
                      G.nodes[node][const.GNodeAttribs.Lng.name],
                      G.nodes[node][const.GNodeAttribs.totDeg.name],
                      G.nodes[node][const.GNodeAttribs.nLines.name],
                      G.nodes[node][const.GNodeAttribs.nServices.name]] for node in G.nodes()]
    dfNdAttribs = pd.DataFrame(nodeStrengths, columns=[const.GNodeAttribs.StationId.name,
                                                       const.GNodeAttribs.Lat.name,
                                                       const.GNodeAttribs.Lng.name,
                                                       const.GNodeAttribs.totDeg.name,
                                                       const.GNodeAttribs.nLines.name,
                                                       const.GNodeAttribs.nServices.name])

    folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)

    ndAttrib = const.GNodeAttribs.totDeg
    myMap = plotNodeFeature(G, dfNdAttribs, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.nLines
    myMap = plotNodeFeature(G, dfNdAttribs, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.nServices
    myMap = plotNodeFeature(G, dfNdAttribs, ndAttrib, myMap)

    '''
    for index, route in dfRoutes.iterrows():
        designedTtb = route[const.RoutesCols.designedTtb.name]
        tupleList = list(zip(designedTtb[const.RoutesTtbCols.StationOrder.name],
                             designedTtb[const.RoutesTtbCols.StationCode.name]))
        dfDesignedTtb = pd.DataFrame(tupleList, columns=[const.RoutesTtbCols.StationOrder.name,
                                                         const.RoutesTtbCols.StationCode.name])
        dfDesignedTtb = dfDesignedTtb.set_index(const.RoutesTtbCols.StationOrder.name)
        dfDesignedTtb.sort_index()

        plotRoutesToLayer(G, dfDesignedTtb, netLayer)
    '''

    LayerControl().add_to(myMap)

    myMap.save(mapfile)

# ======================================================================================================================
def plotClosenessCentralities(G, myMap, mapfile):
    nodeCentrals = [[node,
                     G.nodes[node][const.GNodeAttribs.Lat.name],
                     G.nodes[node][const.GNodeAttribs.Lng.name],
                     G.nodes[node][const.GNodeAttribs.ccHops.name],
                     G.nodes[node][const.GNodeAttribs.ccDist.name],
                     G.nodes[node][const.GNodeAttribs.ccTime.name]] for node in G.nodes()]
    dfNdCentrals = pd.DataFrame(nodeCentrals, columns=[const.GNodeAttribs.StationId.name,
                                                       const.GNodeAttribs.Lat.name,
                                                       const.GNodeAttribs.Lng.name,
                                                       const.GNodeAttribs.ccHops.name,
                                                       const.GNodeAttribs.ccDist.name,
                                                       const.GNodeAttribs.ccTime.name])

    folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)

    ndAttrib = const.GNodeAttribs.ccHops
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.ccDist
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.ccTime
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    LayerControl().add_to(myMap)
    myMap.save(mapfile)

# ======================================================================================================================
def plotBetweennessCentralities(G, myMap, mapfile):
    nodeCentrals = [[node,
                     G.nodes[node][const.GNodeAttribs.Lat.name],
                     G.nodes[node][const.GNodeAttribs.Lng.name],
                     G.nodes[node][const.GNodeAttribs.cbHops.name],
                     G.nodes[node][const.GNodeAttribs.cbDist.name],
                     G.nodes[node][const.GNodeAttribs.cbTime.name]] for node in G.nodes()]
    dfNdCentrals = pd.DataFrame(nodeCentrals, columns=[const.GNodeAttribs.StationId.name,
                                                       const.GNodeAttribs.Lat.name,
                                                       const.GNodeAttribs.Lng.name,
                                                       const.GNodeAttribs.cbHops.name,
                                                       const.GNodeAttribs.cbDist.name,
                                                       const.GNodeAttribs.cbTime.name])

    folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)

    ndAttrib = const.GNodeAttribs.cbHops
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.cbDist
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    ndAttrib = const.GNodeAttribs.cbTime
    myMap = plotNodeFeature(G, dfNdCentrals, ndAttrib, myMap)

    LayerControl().add_to(myMap)

    myMap.save(mapfile)

# ======================================================================================================================
def plotProximityDensity(G, myMap, mapfile):
    '''
    :param G:
    :param myMap:
    :param mapfile:
    :return:
    '''

    proxDensities = [[node,
                      G.nodes[node][const.GNodeAttribs.Lat.name],
                      G.nodes[node][const.GNodeAttribs.Lng.name],
                      G.nodes[node][const.GNodeAttribs.frV_u30.name],
                      G.nodes[node][const.GNodeAttribs.toV_u30.name],
                      G.nodes[node][const.GNodeAttribs.frV_u60.name],
                      G.nodes[node][const.GNodeAttribs.toV_u60.name],
                      G.nodes[node][const.GNodeAttribs.frV_u90.name],
                      G.nodes[node][const.GNodeAttribs.toV_u90.name],
                      G.nodes[node][const.GNodeAttribs.frV_o90.name],
                      G.nodes[node][const.GNodeAttribs.toV_o90.name],
                      G.nodes[node][const.GNodeAttribs.frV_3060.name],
                      G.nodes[node][const.GNodeAttribs.toV_3060.name],
                      G.nodes[node][const.GNodeAttribs.frV_6090.name],
                      G.nodes[node][const.GNodeAttribs.toV_6090.name]] for node in G.nodes]
    dfProxDensities = pd.DataFrame(proxDensities, columns = [const.GNodeAttribs.StationId.name,
                                                             const.GNodeAttribs.Lat.name,
                                                             const.GNodeAttribs.Lng.name,
                                                             const.GNodeAttribs.frV_u30.name,
                                                             const.GNodeAttribs.toV_u30.name,
                                                             const.GNodeAttribs.frV_u60.name,
                                                             const.GNodeAttribs.toV_u60.name,
                                                             const.GNodeAttribs.frV_u90.name,
                                                             const.GNodeAttribs.toV_u90.name,
                                                             const.GNodeAttribs.frV_o90.name,
                                                             const.GNodeAttribs.toV_o90.name,
                                                             const.GNodeAttribs.frV_3060.name,
                                                             const.GNodeAttribs.toV_3060.name,
                                                             const.GNodeAttribs.frV_6090.name,
                                                             const.GNodeAttribs.toV_6090.name])

    minVal = min(dfProxDensities[const.GNodeAttribs.frV_u30.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_u30.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_u60.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_u60.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_u90.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_u90.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_o90.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_o90.name].min())

    maxVal = max(dfProxDensities[const.GNodeAttribs.frV_u30.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_u30.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_u60.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_u60.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_u90.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_u90.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_o90.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_o90.name].max())

    ndAttribs = [const.GNodeAttribs.frV_u30, const.GNodeAttribs.toV_u30,
                 const.GNodeAttribs.frV_u60, const.GNodeAttribs.toV_u60,
                 const.GNodeAttribs.frV_u90, const.GNodeAttribs.toV_u90,
                 const.GNodeAttribs.frV_o90, const.GNodeAttribs.toV_o90]

    '''
    minVal = min(dfProxDensities[const.GNodeAttribs.frV_u30.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_u30.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_3060.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_3060.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_6090.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_6090.name].min(),
                 dfProxDensities[const.GNodeAttribs.frV_o90.name].min(),
                 dfProxDensities[const.GNodeAttribs.toV_o90.name].min())

    maxVal = max(dfProxDensities[const.GNodeAttribs.frV_u30.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_u30.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_3060.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_3060.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_6090.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_6090.name].max(),
                 dfProxDensities[const.GNodeAttribs.frV_o90.name].max(),
                 dfProxDensities[const.GNodeAttribs.toV_o90.name].max())
    
    ndAttribs = [const.GNodeAttribs.frV_u30,    const.GNodeAttribs.toV_u30,
                 const.GNodeAttribs.frV_3060,   const.GNodeAttribs.toV_3060,
                 const.GNodeAttribs.frV_6090,   const.GNodeAttribs.toV_6090,
                 const.GNodeAttribs.frV_o90,   const.GNodeAttribs.toV_o90]
    '''

    colourmap = cm.linear.Reds_09.scale(minVal, maxVal)
    colourmap.caption = 'Proximity density'

    folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)

    for ndAttrib in ndAttribs:
        #myMap = plotNodeFeature(G, dfProxDensities, ndAttrib, myMap)
        myMap = plotNodeFeature(G, dfProxDensities, ndAttrib, myMap, colourmap)

    myMap.add_child(colourmap)

    LayerControl().add_to(myMap)

    myMap.save(mapfile)

