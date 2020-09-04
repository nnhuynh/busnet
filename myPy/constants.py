from enum import Enum

excludedRoutes = ['route72-1.json', # timetable not available (in routeInfo/route72-1.json)
                  'route70-5.json', # because this route has only 2 stops, 1 at HCMC boundary, 1 outside HCMC.
                  'route61-4.json', # this route has only 2 stops, 1 outside of HCMC
                  'route61-7.json' # this route has only 2 stops, 1 outside of HCMC
                  ]

stopsOutsideHCMC = [7509, 7261, 7217, 7218, 7281, 7210, 7209, 7214, 7591, 7257, 7216, 7208, 7215, 7213, 7280, 7211, 7136, 7220, 7263, 7219, 7212, 7595]

maxWalkDist = 300 # metres
walkSpeed = 1.3 # metres/second - average comfortable walk speed for people of all ages
maxWalkTime = maxWalkDist/walkSpeed # in seconds
maxWaitTime = 60*60 # seconds - max wait time at a bus stop for transfering, assumingly 30 minutes.
dfaultDwellTime = 6 # seconds - the default period from when a bus arrives at a bus stop to when it leaves the stop.


dataFolder = '../data'
trashSite = './trashSite'
outputsFolder = './outputs'
nodeDegDistribFolder = '%s/nodeDegDistrib' % outputsFolder
pathLenDistribFolder = '%s/pathLenDistrib' % outputsFolder
nodeCentralsFolder = '%s/nodeCentrals' % outputsFolder
picklesFolder = '%s/pickles' % outputsFolder
mapPlots = '%s/mapPlots' % outputsFolder
ttgAnalysisFolder = '%s/ttgAnalysis' % outputsFolder

nodeDegPkl = '%s/nodeDeg.pkl' % picklesFolder
shortestPathPkl = '%s/shortestHops.pkl' % picklesFolder
shortestTimePkl = '%s/shortestTime.pkl' % picklesFolder
shortestDistPkl = '%s/shortestDist.pkl' % picklesFolder


class RouteInfoCols(Enum):
    routeId = 0
    routeNo = 1
    routeName = 2
    timeTableIn = 3
    timeTableOut = 4

# ======================================================================================================================
class BusMapRouteCols(Enum):
    # Not all columns in the Busmap json file need to be listed here. List only those we need.
    # However exact column names in the json file should be used.
    StationId = 0
    StationOrder = 1
    StationDirection = 2
    RouteId = 3
    StationName = 6
    pathPoints = 7
    Address = 8
    Lat = 9
    Lng = 10

# ======================================================================================================================
class GNodeAttribs(Enum):
    routes = 0
    StationId = 1
    StationDesc = 2
    Lat = 4
    Lng = 5

    neighbourNodes = 'neighbour nodes'

    nLines = 'Number of lines'
    nServices = 'Number of services'
    inDeg = 8
    outDeg = 9
    totDeg = 'Node degree (total)'

    ccHops = 'Closeness centrality (by number of stops)'
    ccTime = 'Closeness centrality (by mean travel time)'
    ccDist = 'Closeness centrality (by travel distance)'
    cbHops = 'Betweenness centrality (by number of stops)'
    cbTime = 'Betweenness centrality (by mean travel time)'
    cbDist = 'Betweenness centrality (by travel distance)'

    frV_u30 = 'Destination proximity density (under 30 minutes)' # as origin
    frV_u60 = 'Destination proximity density (under 60 minutes)' # as origin
    frV_u90 = 'Destination proximity density (under 90 minutes)' # as origin
    frV_o90 = 'Destination proximity density (over 90 minutes)' # as origin
    frV_3060 = 'Destination proximity density (30-60 minutes)' # as origin
    frV_6090 = 'Destination proximity density (60-90 minutes)' # as origin

    toV_u30 = 'Origin proximity density (under 30 minutes)' # as destination
    toV_u60 = 'Origin proximity density (under 60 minutes)' # as destination
    toV_u90 = 'Origin proximity density (under 90 minutes)' # as destination
    toV_o90 = 'Origin proximity density (over 90 minutes)' # as destination
    toV_3060 = 'Origin proximity density (30-60 minutes)' # as destination
    toV_6090 = 'Origin proximity density (60-90 minutes)' # as destination

    ndfrV_u30 = 'Nodes from this station under 30 minutes'
    ndfrV_u60 = 'Nodes from this station under 60 minutes'
    ndfrV_u90 = 'Nodes from this station under 90 minutes'
    ndfrV_o90 = 'Nodes from this station over 90 minutes'
    ndfrV_3060 = 'Nodes from this station 30-60 minutes'
    ndfrV_6090 = 'Nodes from this station 60-90 minutes'

    ndtoV_u30 = 'Nodes to this station under 30 minutes'
    ndtoV_u60 = 'Nodes to this station under 60 minutes'
    ndtoV_u90 = 'Nodes to this station under 90 minutes'
    ndtoV_o90 = 'Nodes to this station over 90 minutes'
    ndtoV_3060 = 'Nodes to this station 30-60 minutes'
    ndtoV_6090 = 'Nodes to this station 60-90 minutes'

class GNodeRouteCols(Enum):
    RouteId = 0
    StationDirection = 1
    StationOrder = 2

# ======================================================================================================================
class GEdgeAttribs(Enum):
    routes = 0
    nLines = 1
    meanRoadDist = 2
    meanTravTime = 3
    nServices = 4

class GEdgeRouteCols(Enum):
    RouteId = 0
    StationDirection = 1
    edgeOrder = 2
    pathPoints = 3
    roadDist = 4
    tTrav = 5


# ======================================================================================================================
class RoutesCols(Enum):
    RouteId = 0
    StationDirection = 1
    RouteNo = 2
    RouteName = 3
    tripTime = 4
    tripDist = 5
    designedTtb = 6
    actualTtb = 7

class RoutesTtbCols(Enum):
    StationOrder = 0
    StationCode = 1
    timetable = 2


# ======================================================================================================================
class GttEdgeAttribs(Enum):
    type = 'type'
    typeBus = 'bus'
    typeWalk = 'walk'
    typeWait = 'wait'
    dirRouteID = 'directed route ID' #e.g 134_0 for routeID 134, direction 0
    time = 'time'
    transfer = 'transfer'

class GttNodeAttribs(Enum):
    ndDesc = 'ndDesc'

class dfNdDetailsCols(Enum):
    RouteId = 'RouteId'
    StationDirection = 'StationDirection'
    StationId = 'StationId'
    time = 'time'
    dirRoute = 'dirRoute'
    arrOrDep = 'arrOrDep'
    arr = 'arr'
    dep = 'dep'
    gttNdID = 'gttNdID'

class dfAllTransfersCols(Enum):
    stopFr = 'stopFr'
    stopTo = 'stopTo'
    nodeFr = 'nodeFr'
    nodeTo = 'nodeTo'
    timeFr = 'timeFr'
    timeTo = 'timeTo'
    type = 'type'