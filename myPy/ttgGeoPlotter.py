import pandas as pd
import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.features import GeoJson, GeoJsonTooltip
from folium.plugins import MarkerCluster
import branca.colormap as cm

import constants as const

def plotProximityDensity(dfNdDetails):
    '''
    :param dfNdDetails:
    :return:
    '''
    pass
    #colourmap = cm.linear.Reds_09.scale(minVal, maxVal)
    #colourmap.caption = 'Proximity density'

    #folium.GeoJson('%s/hcmc.geojson' % const.dataFolder, control=False, show=True, name='HCMC').add_to(myMap)