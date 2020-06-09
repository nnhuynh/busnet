import matplotlib.pyplot as plt
import constants as const
import utilities as utils

xAxisTimestep = 5*60 # 5 minutes
transitStyle = {'linestyle': '-', 'color':'blue', 'linewidth':1, 'alpha':1}
waitStyle = {'linestyle': '--', 'color':'red', 'linewidth':1, 'alpha':0.5}
walkStyle = {'linestyle': '--', 'color':'green', 'linewidth':1, 'alpha':0.5}
gridStyle = {'linestyle': ':', 'color':'black', 'linewidth':0.5, 'alpha':1}

# ======================================================================================================================
def plotWalkLinks(GtopoTemp, inode, jnode, filename):
    for edge in GtopoTemp.edges:
        if (str(inode) in edge[0]) and (str(jnode) in edge[1]) and \
                (GtopoTemp.edges[edge][const.GttEdgeAttribs.type.name]==const.GttEdgeAttribs.typeWalk.name):
            tFr = int(edge[0].split('_')[3])
            tTo = int(edge[1].split('_')[3])
            ndFr = edge[0].split('_')[2]
            ndTo = edge[1].split('_')[2]
            plt.plot([tFr, tTo], [ndFr, ndTo], '--', color='green', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.clf()

# ======================================================================================================================
def plotWaitLinks(GtopoTemp, node, filename):
    '''
    :param GtopoTemp:
    :param node: stationCode in integer
    :return:
    '''
    for edge in GtopoTemp.edges:
        if (str(node) in edge[0]) and (str(node) in edge[1]) and \
                (GtopoTemp.edges[edge][const.GttEdgeAttribs.type.name]==const.GttEdgeAttribs.typeWait.name):
            tFr = int(edge[0].split('_')[3])
            tTo = int(edge[1].split('_')[3])
            ndFr = edge[0].split('_')[2]
            ndTo = edge[1].split('_')[2]
            plt.plot([tFr, tTo], [ndFr, ndTo], '-', color='red', alpha=0.5, linewidth=4)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #plt.clf()

# ======================================================================================================================
def plotTransitLinks(GtopoTemp, routeID, routeDir, timeWindow, filename):
    fig = plt.figure()
    ax = plt.gca()

    selectedEdges = []

    for edge in GtopoTemp.edges:
        tFr = int(edge[0].split('_')[3])
        tTo = int(edge[1].split('_')[3])
        ndFr = edge[0].split('_')[2]
        ndTo = edge[1].split('_')[2]
        if (GtopoTemp.edges[edge][const.GttEdgeAttribs.dirRouteID.name] == '%d_%d' % (routeID, routeDir)) and \
                ((tFr>=timeWindow[0] and tFr<=timeWindow[1]) or (tTo>=timeWindow[0] and tTo<=timeWindow[1])):
            selectedEdges.append(edge)
            ax.plot([tFr, tTo], [ndFr, ndTo],
                    linestyle = transitStyle['linestyle'], color = transitStyle['color'],
                    linewidth = transitStyle['linewidth'], alpha = transitStyle['alpha'])

    ax.grid(color=gridStyle['color'], linestyle=gridStyle['linestyle'], linewidth=gridStyle['linewidth'],
            alpha=gridStyle['alpha'])
    nxTicks = int((timeWindow[1] - timeWindow[0]) / xAxisTimestep + .5)
    xTicks = [timeWindow[0] + i * xAxisTimestep for i in range(nxTicks)]
    xLabls = [utils.convertSecsToHHMMSS(xTick) for xTick in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabls, rotation=90)

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    return selectedEdges

# ======================================================================================================================
def plotTransitLinks(GtopoTemp, ndIDict, routeID, routeDir, timeWindow, filename):
    fig = plt.figure()
    ax = plt.gca()

    selectedEdges = []

    for edge in GtopoTemp.edges:
        tFr = int(ndIDict[edge[0]].split('_')[3])
        tTo = int(ndIDict[edge[1]].split('_')[3])
        stopFr = ndIDict[edge[0]].split('_')[2]
        stopTo = ndIDict[edge[1]].split('_')[2]
        if (GtopoTemp.edges[edge][const.GttEdgeAttribs.dirRouteID.name] == '%d_%d' % (routeID, routeDir)) and \
                ((tFr>=timeWindow[0] and tFr<=timeWindow[1]) or (tTo>=timeWindow[0] and tTo<=timeWindow[1])):
            selectedEdges.append(edge)
            ax.plot([tFr, tTo], [stopFr, stopTo],
                    linestyle = transitStyle['linestyle'], color = transitStyle['color'],
                    linewidth = transitStyle['linewidth'], alpha = transitStyle['alpha'])

    ax.grid(color=gridStyle['color'], linestyle=gridStyle['linestyle'], linewidth=gridStyle['linewidth'],
            alpha=gridStyle['alpha'])
    nxTicks = int((timeWindow[1] - timeWindow[0]) / xAxisTimestep + .5)
    xTicks = [timeWindow[0] + i * xAxisTimestep for i in range(nxTicks)]
    xLabls = [utils.convertSecsToHHMMSS(xTick) for xTick in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabls, rotation=90)

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    return selectedEdges