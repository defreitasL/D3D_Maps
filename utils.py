import numpy as np
import matplotlib.pyplot as plt
import pyproj
import ipywidgets as wid
import xarray as xr
from map_Features import draw_map, draw_HsField, draw_TpField, draw_DirVector

plt.rcParams.update({'font.size': 10,
                     'font.weight': 'bold',
                     'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold',
                     'font.family': 'serif'})

class Domain:
    def __init__(self, ds_path):
        ds = xr.open_dataset(ds_path)
        self.ds = ds
        self.epsg = ds.projected_coordinate_system.epsg
        self.geo_epsg = 4326
        self.transform_coordinates(ds)
        self.time = ds.time.values
        self.corners = [np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)] # [xmin, xmax, ymin, ymax]
        ii_nan = np.isnan(self.lon) | np.isnan(self.lat)
        
        self.lon[ii_nan] = (self.corners[0] + self.corners[1]) / 2
        self.lat[ii_nan] = (self.corners[2] + self.corners[3]) / 2

        ii_nan = np.isnan(self.ds.x.values) | np.isnan(self.ds.y.values) | np.isinf(self.ds.x.values) | np.isinf(self.ds.y.values)
        x = ds.x.values
        y = ds.y.values

        x[ii_nan] = (self.corners[0] + self.corners[1]) / 2
        y[ii_nan] = (self.corners[2] + self.corners[3]) / 2

        n,m = ds.x.shape

        self.Hs = ds.hsign.values
        self.Hs = np.zeros((len(self.ds.time.values), n, m))
        for i in range(len(self.ds.time.values)):
            self.Hs[i,:, :] = ds.hsign[i,:,:].values
            self.Hs[i, ii_nan] = np.nan

        self.Tp = np.zeros((len(self.ds.time.values), n, m))
        for i in range(len(self.ds.time.values)):
            self.Tp[i,:, :] = ds.period[i,:,:].values
            self.Tp[i, ii_nan] = np.nan

        self.dir = np.zeros((len(self.ds.time.values), n, m))
        for i in range(len(self.ds.time.values)):
            self.dir[i,:, :] = ds.dir[i,:,:].values
            self.dir[i, ii_nan] = np.nan

        self.x = x
        self.y = y

    def transform_coordinates(self, ds):
        '''Transform the coordinates to WGS84'''

        gcs = pyproj.CRS.from_epsg(self.geo_epsg)
        utm = pyproj.CRS.from_epsg(self.epsg)

        gcs_t =  pyproj.Transformer.from_crs(utm, gcs, always_xy=True).transform

        empty_x = np.empty_like(ds.x.values)
        empty_y = np.empty_like(ds.y.values)

        ii_nan = np.isnan(ds.x.values) | np.isnan(ds.y.values) | np.isinf(ds.x.values) | np.isinf(ds.y.values)

        empty_x[~ii_nan], empty_y[~ii_nan] = gcs_t(ds.x.values[~ii_nan], ds.y.values[~ii_nan])

        empty_x[ii_nan] = np.nan
        empty_y[ii_nan] = np.nan

        self.lon, self.lat = empty_x, empty_y     
 
    
    def map(self, cfg):

        if cfg['coord_sys'] == 'geographical':
            self.corners = [np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)] # [xmin, xmax, ymin, ymax]
            self.fig, self.ax = draw_map(self.geo_epsg,
                                         self.corners,
                                          cfg['ticks_interval'],
                                          cfg['draw_north_arrow'],
                                          cfg['draw_EPSG'],
                                          cfg['grid_id'])
        else:
            self.corners = [np.nanmin(self.x), np.nanmax(self.x), np.nanmin(self.y), np.nanmax(self.y)]
            self.fig, self.ax = draw_map(self.epsg,
                                         self.corners,
                                         cfg['ticks_interval'],
                                         cfg['draw_north_arrow'],
                                         cfg['draw_EPSG'],
                                         cfg['grid_id'])
        

    def HsField(self, cfg):
        return draw_HsField(self.ax, self.lon, self.lat, self.Hs[cfg["time_id"],:,:], cfg['colorbar_id'], cfg['colormap_alpha'])
    def TpField(self, cfg):
        return draw_TpField(self.ax, self.lon, self.lat, self.Tp[cfg["time_id"],:,:], cfg['colorbar_id'], cfg['colormap_alpha'])
    def DirVector(self, cfg):
        return draw_DirVector(self.ax, self.lon, self.lat, self.dir[cfg["time_id"],:,:], self.Hs[cfg["time_id"],:,:])

class widget_map():
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.domain = Domain(ds_path)

        my_widgets = {}
        my_widgets["var_id"] = wid.Dropdown(
                                options=[(r'Hs', 1),
                                        (r'Tp', 2),
                                        (r'$H_{s}$', 3),
                                        (r'$Wind vector$', 4),
                                        (r'$Current vector$', 5)
                                        ],
                                value=1,
                                description='Variable to plot:')
        my_widgets["na_id"] = wid.Checkbox(
                            value=True,
                            description='North arrow',
                            disabled=False,
                            indent=False
                            )
        my_widgets["epsg_id"] = wid.Checkbox(
                            value=True,
                            description='EPSG code',
                            disabled=False,
                            indent=False
                            )
        my_widgets["coord_sys"] = wid.RadioButtons(
        options=['geographical', 'projected'],
        value='geographical', # Defaults to 'pineapple'
    #    layout={'width': 'max-content'}, # If the items' names are long
        description='Coordinates system:',
        disabled=False
        )
        if my_widgets["coord_sys"].value == 'geographical':
            my_widgets["ticks_interval"] = wid.FloatSlider(
            value=0.25,
            min=0.01,
            max=1.0,
            step=0.01,
            description='Ticks interval:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            )
        else:
            my_widgets["ticks_interval"] = wid.FloatSlider(
            value=1000,
            min=100,
            max=10000,
            step=100,
            description='Ticks interval:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            )
    

        time_options = get_timelist(self.domain.time)

        my_widgets["time_id"] = wid.Dropdown(
            options=time_options,
            value=0,
            description='Time to plot:')
        
        my_widgets["colorbar_id"] = wid.Dropdown(options=cmaps_list(),
                                                 value=79, 
                                                 description='Colormap',
                                                 indent=False)

        my_widgets["colormap_alpha"] = wid.FloatSlider(
            value=1,
            min=0,
            max=1,
            step=0.1,
            description='Transparency:')
        
        my_widgets["dir_vector_id"] = wid.Checkbox(
                            value=False,
                            description='Direction vector',
                            disabled=False,
                            indent=False
                            )
        

        
        my_widgets["map_setup_id"] = wid.Button(description='Update map')
        self.out = wid.Output()

        my_widgets["grid_id"] = wid.Checkbox(
                            value=True, 
                            description='Grid',
                            disabled=False,
                            indent=False)

        c_1 = wid.VBox([my_widgets["var_id"],
                         my_widgets["ticks_interval"],
                         my_widgets["dir_vector_id"],
                         my_widgets["na_id"],
                          my_widgets["coord_sys"]])
        c_2 = wid.VBox([my_widgets["time_id"],
                        my_widgets["colormap_alpha"],
                        my_widgets["grid_id"],
                        my_widgets["epsg_id"],
                        my_widgets["colorbar_id"],
                        my_widgets["map_setup_id"]])
        controls = wid.HBox([c_1, c_2])
        controls = wid.VBox([controls, self.out])
        display(controls)
        
        self.my_widgets = my_widgets

        my_widgets["map_setup_id"].on_click(self.update_map)

        cfg = {
            'draw_north_arrow': self.my_widgets["na_id"].value,
            'draw_EPSG': self.my_widgets["epsg_id"].value,
            'var_id': self.my_widgets["var_id"].value,
            'time_id': self.my_widgets["time_id"].value,
            'ticks_interval': self.my_widgets["ticks_interval"].value,
            'coord_sys': self.my_widgets["coord_sys"].value,
            'colorbar_id': cmaps_list()[self.my_widgets["colorbar_id"].value][0],
            'grid_id': self.my_widgets["grid_id"].value,
            'colormap_alpha': self.my_widgets["colormap_alpha"].value,        
        }
        self.domain.map(cfg)
        self.domain.HsField(cfg)
        with self.out:
            plt.show()
            self.domain.fig.canvas.draw()
            self.domain.fig.canvas.toolbar_visible = True

    def update_map(self, b):

        cfg = {
            'draw_north_arrow': self.my_widgets["na_id"].value,
            'draw_EPSG': self.my_widgets["epsg_id"].value,
            'var_id': self.my_widgets["var_id"].value,
            'time_id': self.my_widgets["time_id"].value,
            'ticks_interval': self.my_widgets["ticks_interval"].value,
            'coord_sys': self.my_widgets["coord_sys"].value,
            'colorbar_id': cmaps_list()[self.my_widgets["colorbar_id"].value][0],
            'grid_id': self.my_widgets["grid_id"].value,
            'colormap_alpha': self.my_widgets["colormap_alpha"].value         
        }

        self.domain.ax.clear()
        self.domain.map(cfg)
        if cfg['var_id'] == 1:
            self.domain.HsField(cfg)
        else:
            self.domain.TpField(cfg)
        if self.my_widgets["dir_vector_id"].value:
            self.domain.DirVector(cfg)


        with self.out:
            self.out.clear_output()
            plt.show()

        self.domain.fig.canvas.draw()
        self.domain.fig.canvas.toolbar_visible = True

def get_timelist(time):
    
    return [(str(t), i) for i, t in enumerate(time)]

def cmaps_list():
    '''List of colormaps'''
    
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']


    return [(cmap, i) for i, cmap in enumerate(cmaps)]