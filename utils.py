import numpy as np
import matplotlib.pyplot as plt
import pyproj
import ipywidgets as wid
import xarray as xr
from map_Features import draw_map, draw_generic_var, draw_DirVector, draw_var_nodes, draw_var_edges, draw_var_faces

plt.rcParams.update({'font.size': 10,
                     'font.weight': 'bold',
                     'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold',
                     'font.family': 'serif'})

class Domain:
    def __init__(self, ds_path):
        ds = xr.open_dataset(ds_path)
        self.ds = ds
        self.var_list = list(ds.data_vars.keys())
        try:
            self.var_list.remove('projected_coordinate_system')
            self.var_list.remove('kcs')
        except:
            pass
        self.units = [ds[var].units for var in self.var_list]
        self.epsg = ds.projected_coordinate_system.epsg
        self.geo_epsg = 4326
        self.transform_coordinates(ds)
        self.time = ds.time.values
        self.corners = [np.nanmin(self.lon['mesh2d_node']), np.nanmax(self.lon['mesh2d_node']), np.nanmin(self.lat['mesh2d_node']), np.nanmax(self.lat['mesh2d_node'])] # [xmin, xmax, ymin, ymax]
        ii_nan = np.isnan(self.lon) | np.isnan(self.lat)
        
        self.lon[ii_nan] = (self.corners[0] + self.corners[1]) / 2
        self.lat[ii_nan] = (self.corners[2] + self.corners[3]) / 2

        ii_nan = np.isnan(self.ds.x.values) | np.isnan(self.ds.y.values) | np.isinf(self.ds.x.values) | np.isinf(self.ds.y.values)
        x = ds.x.values
        y = ds.y.values

        n,m = ds.x.shape

        self.x = x.reshape(n, m)
        self.y = y.reshape(n, m)
        self.ii_nan = ii_nan

    def get_var(self, var_id):
        '''Get the variable to plot'''
        var_id = self.var_list[var_id]

        n,m = self.ds.x.shape

        plot_var = np.zeros((len(self.ds.time.values), n, m))
        for i in range(len(self.ds.time.values)):
            plot_var[i,:, :] = self.ds[var_id][i,:,:].values
            plot_var[i, self.ii_nan] = np.nan

        return plot_var

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
    def plotter(self, cfg):
        var = self.get_var(cfg['var_id'])
        return draw_generic_var(self.ax, self.lon, self.lat, var[cfg["time_id"],:,:], cfg['colorbar_id'], cfg['colormap_alpha'], var_id = self.var_list[cfg['var_id']], units = self.units[cfg['var_id']]) 

    # def HsField(self, cfg):
    #     return draw_HsField(self.ax, self.lon, self.lat, self.Hs[cfg["time_id"],:,:], cfg['colorbar_id'], cfg['colormap_alpha'])
    # def TpField(self, cfg):
    #     return draw_TpField(self.ax, self.lon, self.lat, self.Tp[cfg["time_id"],:,:], cfg['colorbar_id'], cfg['colormap_alpha'])
    def DirVector(self, cfg):
        dir = self.get_var(self.var_list.index('dir'))
        hs = self.get_var(self.var_list.index('hsign'))
        return draw_DirVector(self.ax, self.lon, self.lat, dir[cfg["time_id"],:,:], hs[cfg["time_id"],:,:])

    
class widget_map():
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.domain = Domain(ds_path)

        self.my_widgets = {}
        self.my_widgets["var_id"] = wid.Dropdown(
            options=[(self.domain.var_list[i], i) for i in range(len(self.domain.var_list))],
            value=0,
            description='Variable to plot:'
        )
        self.my_widgets["na_id"] = wid.Checkbox(
            value=True,
            description='North arrow',
            disabled=False,
            indent=False
        )
        self.my_widgets["epsg_id"] = wid.Checkbox(
            value=True,
            description='EPSG code',
            disabled=False,
            indent=False
        )
        self.my_widgets["coord_sys"] = wid.RadioButtons(
            options=['geographical', 'projected'],
            value='geographical',  # Defaults to 'geographical'
            description='Coordinates system:',
            disabled=False
        )

        # Initialize ticks_interval widget
        self.my_widgets["ticks_interval"] = wid.FloatSlider(
            value=0.25,
            min=0.01,
            max=1.0,
            step=0.01,
            description='Ticks interval:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f'
        )

        # Set up a listener for coord_sys changes
        self.my_widgets["coord_sys"].observe(self.update_ticks_interval, names='value')

        time_options = get_timelist(self.domain.time)

        self.my_widgets["time_id"] = wid.Dropdown(
            options=time_options,
            value=0,
            description='Time to plot:'
        )

        self.my_widgets["colorbar_id"] = wid.Dropdown(options=cmaps_list(),
                                                      value=79,
                                                      description='Colormap',
                                                      indent=False)

        self.my_widgets["colormap_alpha"] = wid.FloatSlider(
            value=1,
            min=0,
            max=1,
            step=0.1,
            description='Transparency:'
        )

        self.my_widgets["dir_vector_id"] = wid.Checkbox(
            value=False,
            description='Direction vector',
            disabled=False,
            indent=False
        )

        self.my_widgets["map_setup_id"] = wid.Button(description='Update map')
        self.out = wid.Output()

        self.my_widgets["grid_id"] = wid.Checkbox(
            value=True,
            description='Grid',
            disabled=False,
            indent=False)

        c_1 = wid.VBox([self.my_widgets["var_id"],
                        self.my_widgets["ticks_interval"],
                        self.my_widgets["dir_vector_id"],
                        self.my_widgets["na_id"],
                        self.my_widgets["coord_sys"]])
        c_2 = wid.VBox([self.my_widgets["time_id"],
                        self.my_widgets["colormap_alpha"],
                        self.my_widgets["grid_id"],
                        self.my_widgets["epsg_id"],
                        self.my_widgets["colorbar_id"],
                        self.my_widgets["map_setup_id"]])
        controls = wid.HBox([c_1, c_2])
        controls = wid.VBox([controls, self.out])
        display(controls)

        self.my_widgets["map_setup_id"].on_click(self.update_map)

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
        self.domain.plotter(cfg)
        with self.out:
            plt.show()
            self.domain.fig.canvas.draw()
            self.domain.fig.canvas.toolbar_visible = True

    def update_ticks_interval(self, change):
        if change['new'] == 'geographical':
            # Ajustar o valor temporário primeiro
            self.my_widgets["ticks_interval"].value = 0.25
            self.my_widgets["ticks_interval"].min = 0.01
            self.my_widgets["ticks_interval"].max = 3.0
            self.my_widgets["ticks_interval"].step = 0.05
            self.my_widgets["ticks_interval"].value = 0.25
        else:
            # Ajustar o valor temporário primeiro
            self.my_widgets["ticks_interval"].value = 10000
            self.my_widgets["ticks_interval"].max = 50000
            self.my_widgets["ticks_interval"].min = 1000
            self.my_widgets["ticks_interval"].step = 500
            self.my_widgets["ticks_interval"].value = 10000

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

        self.domain.fig.clear()
        self.domain.map(cfg)
        
        self.domain.plotter(cfg)


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

def check_epsg(fc):
    '''Check if the EPSG code is valid'''

    ds = xr.open_dataset(fc)
    if ds.projected_coordinate_system.epsg == 0:
        bool_epsg = False
    else:
        bool_epsg = True

    return bool_epsg

def set_epsg(fc, epsg):
    '''Set the EPSG code in the dataset'''

    ds = xr.open_dataset(fc)
    ds.attrs['projected_coordinate_system'] = epsg
    ds.to_netcdf(fc)
    ds.close()

class Domain_FM:
    def __init__(self, ds_path):
        ds = xr.open_dataset(ds_path)
        self.ds = ds
        self.var_list = list(ds.data_vars.keys())
        self.coord_list = [list(ds[key].coords) for key in ds.data_vars.keys()]
        self.remove_non_var()
        self.longname_var()
        self.get_units()
        self.epsg = ds.projected_coordinate_system.epsg
        self.geo_epsg = 4326
        self.transform_coordinates(ds)
        self.time = ds.time.values

    def get_var(self, var_id):
        '''Get the variable to plot'''

        var_id = self.var_list[var_id]

        plot_var = self.ds[var_id].values

        return plot_var

    def transform_coordinates(self, ds):
        '''Transform the coordinates to WGS84'''

        gcs = pyproj.CRS.from_epsg(self.geo_epsg)
        utm = pyproj.CRS.from_epsg(self.epsg)

        gcs_t =  pyproj.Transformer.from_crs(utm, gcs, always_xy=True).transform

        coords= []
        for key in list(ds.coords.keys()):
            coords.append(key[:-2])

        self.lon = {}
        self.lat = {}

        for key in np.unique(coords):
            if key != 'ti':
                
                xx = ds[key+'_x'].values
                yy = ds[key+'_y'].values

                empty_x = np.empty_like(xx)
                empty_y = np.empty_like(yy)

                ii_nan = np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy)

                empty_x[~ii_nan], empty_y[~ii_nan] = gcs_t(xx[~ii_nan], yy[~ii_nan])

                empty_x[ii_nan] = np.nan
                empty_y[ii_nan] = np.nan 

                self.lon[key], self.lat[key] = empty_x, empty_y  
    
    def map(self, cfg):

        if cfg['coord_sys'] == 'geographical':

            self.get_corners(self.coord_list[cfg['var_id']])

            self.fig, self.ax = draw_map(self.geo_epsg,
                                         self.corners,
                                          cfg['ticks_interval'],
                                          cfg['draw_north_arrow'],
                                          cfg['draw_EPSG'],
                                          cfg['grid_id'])
        else:
            self.get_corners_utm(self.coord_list[cfg['var_id']])
            self.fig, self.ax = draw_map(self.epsg,
                                         self.corners,
                                         cfg['ticks_interval'],
                                         cfg['draw_north_arrow'],
                                         cfg['draw_EPSG'],
                                         cfg['grid_id'])
    
    def plotter_nodes(self, cfg):
        var = self.get_var(cfg['var_id'])

        faces = self.ds['mesh2d_face_nodes'].values - 1  # 1-based para 0-based

        return draw_var_nodes(self.ax, self.lon['mesh2d_node'], self.lat['mesh2d_node'], faces, var[cfg["time_id"],:], cfg['colorbar_id'], cfg['colormap_alpha'], var_id = self.var_list[cfg['var_id']], units = self.units[cfg['var_id']])
    
    def plotter_edges(self, cfg):
        var = self.get_var(cfg['var_id'])

        edges = self.ds['mesh2d_edge_nodes'].values - 1  # 1-based para 0-based

        return draw_var_edges(self.ax, self.lon['mesh2d_edge'], self.lat['mesh2d_edge'], edges, var[cfg["time_id"],:], cfg['colorbar_id'], cfg['colormap_alpha'], var_id = self.var_list[cfg['var_id']], units = self.units[cfg['var_id']])
    
    def plotter_faces(self, cfg):
        var = self.get_var(cfg['var_id'])

        faces = self.ds['mesh2d_face_nodes'].values - 1

        return draw_var_faces(self.ax, self.lon['mesh2d_face'], self.lat['mesh2d_face'], faces, var[cfg["time_id"],:], cfg['colorbar_id'], cfg['colormap_alpha'], var_id = self.var_list[cfg['var_id']], units = self.units[cfg['var_id']])
    

    def DirVector(self, cfg):
        dir = self.get_var(self.var_list.index('dir'))
        hs = self.get_var(self.var_list.index('hsign'))
        return draw_DirVector(self.ax, self.lon, self.lat, dir[cfg["time_id"],:,:], hs[cfg["time_id"],:,:])
    
    def longname_var(self):
        self.long_name_list = []
        for key in self.var_list:
            try:
                self.long_name_list.append(self.ds[key].long_name)
            except:
                self.long_name_list.append('-1')

        ii = np.where(np.array(self.long_name_list) != '-1')[0]
        self.long_name_list = [self.long_name_list[i] for i in ii]
        self.var_list = [self.var_list[i] for i in ii]
        self.coord_list = [self.coord_list[i] for i in ii]
        
    def remove_non_var(self):
        bool_var = []
        x = False
        for coord in self.coord_list:
            for vc in valid_coords:
                if vc in coord:
                    x = True
 
            if x:           
                bool_var.append(True)
            else:
                bool_var.append(False)
            
            x = False

        self.var_list = [self.var_list[i] for i in range(len(self.var_list)) if bool_var[i]]
        self.coord_list = [self.coord_list[i] for i in range(len(self.coord_list)) if bool_var[i]]

    def get_corners(self, coords):
        '''Get the corners of the domain'''
        for coord in coords:
            if coord in valid_coords:
                crd = coord[:-2]
                self.corners = [np.nanmin(self.lon[crd]), np.nanmax(self.lon[crd]), np.nanmin(self.lat[crd]), np.nanmax(self.lat[crd])] # [xmin, xmax, ymin, ymax] 
                break
    
    def get_corners_utm(self, coords):
        '''Get the corners of the domain'''
        for coord in coords:
            if coord in valid_coords:
                crd = coord[:-2]
                self.corners = [np.nanmin(self.ds[crd+'_x']), np.nanmax(self.ds[crd+'_x']), np.nanmin(self.ds[crd+'_y']), np.nanmax(self.ds[crd+'_y'])] # [xmin, xmax, ymin, ymax] 
                break

    def get_coords(self, coords):
        for coord in coords:
            if coord in valid_coords:
                return coord
            
    def get_units(self):
        '''Get the units of the variables'''

        self.units = []
        for var in self.var_list:
            try:
                self.units.append(self.ds[var].units)
            except:
                self.units.append('-')



class widget_map_FM():
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.domain = Domain_FM(ds_path)

        self.my_widgets = {}
        self.my_widgets["var_id"] = wid.Dropdown(
            options=[(self.domain.long_name_list[i], i) for i in range(len(self.long_name_list))],
            value=0,
            description='Variable to plot:'
        )
        self.my_widgets["na_id"] = wid.Checkbox(
            value=True,
            description='North arrow',
            disabled=False,
            indent=False
        )
        self.my_widgets["epsg_id"] = wid.Checkbox(
            value=True,
            description='EPSG code',
            disabled=False,
            indent=False
        )
        self.my_widgets["coord_sys"] = wid.RadioButtons(
            options=['geographical', 'projected'],
            value='geographical',  # Defaults to 'geographical'
            description='Coordinates system:',
            disabled=False
        )

        # Initialize ticks_interval widget
        self.my_widgets["ticks_interval"] = wid.FloatSlider(
            value=0.25,
            min=0.01,
            max=1.0,
            step=0.01,
            description='Ticks interval:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f'
        )

        # Set up a listener for coord_sys changes
        self.my_widgets["coord_sys"].observe(self.update_ticks_interval, names='value')

        time_options = get_timelist(self.domain.time)

        self.my_widgets["time_id"] = wid.Dropdown(
            options=time_options,
            value=0,
            description='Time to plot:'
        )

        self.my_widgets["colorbar_id"] = wid.Dropdown(options=cmaps_list(),
                                                      value=79,
                                                      description='Colormap',
                                                      indent=False)

        self.my_widgets["colormap_alpha"] = wid.FloatSlider(
            value=1,
            min=0,
            max=1,
            step=0.1,
            description='Transparency:'
        )

        self.my_widgets["dir_vector_id"] = wid.Checkbox(
            value=False,
            description='Direction vector',
            disabled=False,
            indent=False
        )

        self.my_widgets["map_setup_id"] = wid.Button(description='Update map')
        self.out = wid.Output()

        self.my_widgets["grid_id"] = wid.Checkbox(
            value=True,
            description='Grid',
            disabled=False,
            indent=False)

        c_1 = wid.VBox([self.my_widgets["var_id"],
                        self.my_widgets["ticks_interval"],
                        self.my_widgets["dir_vector_id"],
                        self.my_widgets["na_id"],
                        self.my_widgets["coord_sys"]])
        c_2 = wid.VBox([self.my_widgets["time_id"],
                        self.my_widgets["colormap_alpha"],
                        self.my_widgets["grid_id"],
                        self.my_widgets["epsg_id"],
                        self.my_widgets["colorbar_id"],
                        self.my_widgets["map_setup_id"]])
        controls = wid.HBox([c_1, c_2])
        controls = wid.VBox([controls, self.out])
        display(controls)

        self.my_widgets["map_setup_id"].on_click(self.update_map)

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
        self.domain.plotter(cfg)
        with self.out:
            plt.show()
            self.domain.fig.canvas.draw()
            self.domain.fig.canvas.toolbar_visible = True

    def update_ticks_interval(self, change):
        if change['new'] == 'geographical':
            # Ajustar o valor temporário primeiro
            self.my_widgets["ticks_interval"].value = 0.25
            self.my_widgets["ticks_interval"].min = 0.01
            self.my_widgets["ticks_interval"].max = 3.0
            self.my_widgets["ticks_interval"].step = 0.05
            self.my_widgets["ticks_interval"].value = 0.25
        else:
            # Ajustar o valor temporário primeiro
            self.my_widgets["ticks_interval"].value = 10000
            self.my_widgets["ticks_interval"].max = 50000
            self.my_widgets["ticks_interval"].min = 1000
            self.my_widgets["ticks_interval"].step = 500
            self.my_widgets["ticks_interval"].value = 10000

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

        self.domain.fig.clear()
        self.domain.map(cfg)
        
        self.domain.plotter(cfg)


        if self.my_widgets["dir_vector_id"].value:
            self.domain.DirVector(cfg)

        with self.out:
            self.out.clear_output()
            plt.show()

        self.domain.fig.canvas.draw()
        self.domain.fig.canvas.toolbar_visible = True


valid_coords = ['mesh2d_edge_x',
                'mesh2d_edge_y',
                'mesh2d_node_x',
                'mesh2d_node_y',
                'mesh2d_face_x',
                'mesh2d_face_y']