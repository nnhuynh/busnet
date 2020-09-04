import matplotlib.pyplot as plt
import pandas as pd
import constants as const
import utilities as utils

xAxisTimestep = 30*60 # 5 minutes
transitStyle = {'linestyle': '-', 'color':'blue', 'linewidth':1, 'alpha':1}
waitStyle = {'linestyle': '--', 'color':'red', 'linewidth':1, 'alpha':0.5}
walkStyle = {'linestyle': '--', 'color':'green', 'linewidth':1, 'alpha':0.5}
gridStyle = {'linestyle': ':', 'color':'black', 'linewidth':0.5, 'alpha':1}

# ======================================================================================================================
def plotTransitLinks(gtt, edgeSet, timeWindow, filename):
    fig = plt.figure()
    ax = plt.gca()

    for edge in edgeSet:
        tFr = int(edge.source_vertex['ndDesc'].split('_')[3])
        tTo = int(edge.target_vertex['ndDesc'].split('_')[3])
        stopFr = edge.source_vertex['ndDesc'].split('_')[2]
        stopTo = edge.target_vertex['ndDesc'].split('_')[2]
        ax.plot([tFr, tTo], [stopFr, stopTo],
                linestyle=transitStyle['linestyle'], color=transitStyle['color'],
                linewidth=transitStyle['linewidth'], alpha=transitStyle['alpha'])
        print('%d, %s - %d, %s' % (tFr, stopFr, tTo, stopTo))


    ax.grid(color=gridStyle['color'], linestyle=gridStyle['linestyle'], linewidth=gridStyle['linewidth'],
            alpha=gridStyle['alpha'])
    nxTicks = int((timeWindow[1] - timeWindow[0]) / xAxisTimestep + .5)
    xTicks = [timeWindow[0] + i * xAxisTimestep for i in range(nxTicks)]
    xLabls = [utils.convertSecsToHHMMSS(xTick) for xTick in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabls, rotation=90)

    mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.savefig(filename)#, bbox_inches='tight', pad_inches=0)

# ======================================================================================================================
def plotTransitLinks_v2(gtt, dirRouteID, timeWindow, filename):
    fig = plt.figure()
    ax = plt.gca()

    edgeSel = []

    edgeSet = gtt.es.select(dirRouteID_in=[dirRouteID])
    for edge in edgeSet:
        sourceNdDesc = edge.source_vertex['ndDesc']
        targetNdDesc = edge.target_vertex['ndDesc']

        # gets the time at source and target
        sourceTime = int(sourceNdDesc.split('_')[3])
        sourceStop = int(sourceNdDesc.split('_')[2])
        targetTime = int(targetNdDesc.split('_')[3])
        targetStop = int(targetNdDesc.split('_')[2])
        if (sourceTime >= timeWindow[0] and sourceTime <= timeWindow[1]) and \
                (targetTime >= timeWindow[0] and targetTime <= timeWindow[1]):
            ax.plot([sourceTime, targetTime], [sourceStop, targetStop],
                    linestyle=transitStyle['linestyle'], color=transitStyle['color'],
                    linewidth=transitStyle['linewidth'], alpha=transitStyle['alpha'])
            edgeSel.append([edge.source, sourceTime, int(sourceStop),
                            edge.target, targetTime, int(targetStop)])

    ax.grid(color=gridStyle['color'], linestyle=gridStyle['linestyle'], linewidth=gridStyle['linewidth'],
            alpha=gridStyle['alpha'])
    nxTicks = int((timeWindow[1] - timeWindow[0]) / xAxisTimestep + .5)
    xTicks = [timeWindow[0] + i * xAxisTimestep for i in range(nxTicks)]
    xLabls = [utils.convertSecsToHHMMSS(xTick) for xTick in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabls, rotation=90)

    mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.savefig(filename)  # , bbox_inches='tight', pad_inches=0)

    return pd.DataFrame(edgeSel, columns=['srcNdId', 'srcTime', 'srcStop', 'tarNdId', 'tarTime', 'tarStop'])

def plotRoutesAllday(dfRoutes):
    timeGapByRouteByHr = []
    for idx, route in dfRoutes.iterrows():
        routeID = route[const.RoutesCols.RouteId.name]
        routeDir = route[const.RoutesCols.StationDirection.name]
        designedTtb = route[const.RoutesCols.designedTtb.name]
        stnList = designedTtb[const.RoutesTtbCols.StationCode.name]
        ttb = designedTtb[const.RoutesTtbCols.timetable.name]
        for iServ in range(1,len(ttb)):
            timegap = ttb[iServ][0][0] - ttb[iServ-1][0][0]
            timeGapByRouteByHr.append(['%d_%d' % (routeID, routeDir),
                                       utils.convertSecsToHHMMSS(ttb[iServ][0][0]),
                                       timegap])
    df = pd.DataFrame(timeGapByRouteByHr, columns=['dirRouteID', 'timeOfDay', 'timegap'])
    df.to_csv('timeGapByRouteByHr.csv', index=False)