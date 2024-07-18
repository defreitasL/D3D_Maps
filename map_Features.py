import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
import imageio
import os
from matplotlib.patches import FancyArrowPatch
import contextily as cx
from matplotlib.path import Path
import pyproj
import matplotlib.tri as tri

def get_limits(full_run, Obs, full_run_list, rounder):
    '''Get the limits for the plot'''

    
    lower = np.min([np.min(Obs), np.min(full_run)])
    upper = np.max([np.max(Obs), np.max(full_run)])

    if len(full_run_list) > 0:
        for run in full_run_list:
            lower = np.min([lower, np.min(run)])
            upper = np.max([upper, np.max(run)])
    
    lower = np.floor(lower / rounder) * rounder
    upper = np.ceil(upper / rounder) * rounder

    return lower, upper

def get_limits_map(crns, rounder):
    '''Get the limits for the plot'''

    x_lower = np.floor(crns[0] / rounder) * rounder
    x_upper = np.ceil(crns[1] / rounder) * rounder

    y_lower = np.floor(crns[2] / rounder) * rounder
    y_upper = np.ceil(crns[3] / rounder) * rounder

    return [x_lower, x_upper, y_lower, y_upper]

def get_map_ticks(corners, rounder):
    '''Get the ticks for the map'''

    x_lower = np.floor(corners[0] / rounder) * rounder
    x_upper = np.ceil(corners[1] / rounder) * rounder

    y_lower = np.floor(corners[2] / rounder) * rounder
    y_upper = np.ceil(corners[3] / rounder) * rounder

    x_ticks = np.arange(x_lower, x_upper + rounder, rounder)
    y_ticks = np.arange(y_lower, y_upper + rounder, rounder)

    return x_ticks, y_ticks

def draw_arrow_north(ax, pos, corners):
    '''Draw a north arrow'''

    # corners = ax.get_extent()

    xy_tot = [corners[1] - corners[0], corners[3] - corners[2]]

    rel = np.max(xy_tot)

    xy_rel = [xy_tot[0]/rel, xy_tot[1]/rel]

    north = [xy_rel[0]*0.06*rel, xy_rel[1]*0.12*rel]

    despl = [xy_rel[0]*0.013*rel, xy_rel[1]*0.013*rel]

    if pos == 'topright':
        northXYflag = [corners[1] - despl[0] - north[0], corners[3] - despl[1] - north[1]]
    elif pos == 'topleft':
        northXYflag = [corners[0] + despl[0], corners[3] - despl[1] - north[1]]
    elif pos == 'bottomright':
        northXYflag = [corners[1] - despl[0] - north[0], corners[2] + despl[1]]
    elif pos == 'bottomleft':
        northXYflag = [corners[0] + despl[0], corners[2] + despl[1]]

    xRose = [northXYflag[0]+north[0]/2,
             northXYflag[0]+north[0],
             northXYflag[0]+north[0]/2,
             northXYflag[0],
             northXYflag[0]+north[0]/2]
    yRose = [northXYflag[1]+north[1],
             northXYflag[1],
             northXYflag[1]+north[1]/3,
             northXYflag[1],
             northXYflag[1]+north[1]]
    ax.fill(xRose, yRose, color='white', edgecolor='k', linewidth=1.5, zorder=10)
    ax.plot([xRose[0],xRose[2]],[yRose[0],yRose[2]], color = 'k', lw = 1, ls='-', zorder=11)

def draw_EPSG(ax, EPSG, pos, corners):
    '''Draw the EPSG code'''

    # corners = ax.get_extent()


    EPSGdict = dict(boxstyle="round",
                    edgecolor="black",
                    fc = "white")

    desplEPSG = [0.13*(corners[1] - corners[0]), 0.04*(corners[3] - corners[2])]
    text_in = r'EPSG: '+str(int(EPSG))
    if pos == 'topright':
        ax.text(ax.get_xlim()[1]-desplEPSG[0], ax.get_ylim()[1]-desplEPSG[1],
            text_in, bbox=EPSGdict, fontweight='bold', verticalalignment='center', horizontalalignment='center', size = 8)
    elif pos == 'topleft':
        ax.text(ax.get_xlim()[0]+desplEPSG[0], ax.get_ylim()[1]-desplEPSG[1],
            text_in, bbox=EPSGdict, fontweight='bold', verticalalignment='center', horizontalalignment='center', size = 8)
    elif pos == 'bottomright':
        ax.text(ax.get_xlim()[1]-desplEPSG[0], ax.get_ylim()[0]+desplEPSG[1],
            text_in, bbox=EPSGdict, fontweight='bold', verticalalignment='center', horizontalalignment='center', size = 8)
    elif pos == 'bottomleft':
        ax.text(ax.get_xlim()[0]+desplEPSG[0], ax.get_ylim()[0]+desplEPSG[1],
            text_in, bbox=EPSGdict, fontweight='bold', verticalalignment='center', horizontalalignment='center', size = 8)

def draw_map(EPSG, corners, rounder, grid_bool, north_arrow_bool, EPSG_bool):
    '''Draw the map'''

    corners_geo = get_limits_map(corners, rounder)
    epsg_obj = ccrs.PlateCarree()

    if EPSG != 4326:
        gcs = pyproj.CRS.from_epsg(4326)
        utm = pyproj.CRS.from_epsg(EPSG)
        gcs_t = pyproj.Transformer.from_crs(utm, gcs, always_xy=True).transform
        x1, y1 = gcs_t(corners_geo[0], corners_geo[2])
        x2, y2 = gcs_t(corners_geo[1], corners_geo[3])
        corners_geo = [x1, x2, y1, y2]
        x_ticks, y_ticks = get_map_ticks(corners, rounder)
        x_ticks_label = ["{:.0f}".format(k) for k in x_ticks]
        y_ticks_label = ["{:.0f}".format(k) for k in y_ticks]
        x_ticks, _ = gcs_t(x_ticks, np.zeros_like(x_ticks))
        _, y_ticks = gcs_t(np.zeros_like(y_ticks), y_ticks)
    else:
        x_ticks, y_ticks = get_map_ticks(corners, rounder)
        x_ticks_label = ["{:.2f}".format(k) for k in x_ticks]
        y_ticks_label = ["{:.2f}".format(k) for k in y_ticks]



    fh = (corners_geo[1] - corners_geo[0])
    fw = (corners_geo[3] - corners_geo[2])

    fh = fh / np.max([fh, fw])
    fw = fw / np.max([fh, fw])

    fig = plt.figure(figsize=(12*fh, 8*fw), dpi = 300)
    ax = fig.add_subplot(1, 1, 1, projection=epsg_obj)

    ax.set_xlim([corners_geo[0], corners_geo[1]])
    ax.set_ylim([corners_geo[2], corners_geo[3]])

    cx.add_basemap(ax, crs=epsg_obj,
                   source=cx.providers.Esri.WorldImagery,
                   reset_extent=False)

    ax.set_xlim([corners_geo[0], corners_geo[1]])
    ax.set_ylim([corners_geo[2], corners_geo[3]])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_label, rotation=45)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_label, rotation=45)

    if EPSG == 4326:
        ax.set_xlabel(r'$lon [\degree]$')
        ax.set_ylabel(r'$lat [\degree]$')
    else:
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

    if grid_bool:
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.grid(False)

    if north_arrow_bool:
        draw_arrow_north(ax, 'topright', corners_geo)

    if EPSG_bool:
        draw_EPSG(ax, EPSG, 'topleft', corners_geo)

    return fig, ax


def draw_HsField(ax, x, y, Hs, cmap, alpha):

    Hs_upper = np.ceil(np.nanmax(Hs) / 0.5) * 0.5  

    contour = ax.contourf(x, y, Hs, cmap=cmap, extend='max', alpha=alpha, vmin=0, vmax=np.nanmax(Hs), zorder=1, levels = 100, linestyles='none')
    cbar = plt.colorbar(contour, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Significant wave height [m]')
    cbar.set_ticks(np.arange(0, Hs_upper, 0.5))

def draw_TpField(ax, x, y, Tp, cmap, alpha):

    Tp_upper = np.ceil(np.nanmax(Tp) / 1) * 1  

    contour = ax.contourf(x, y, Tp, cmap=cmap, extend='max', alpha=alpha, vmin=0, vmax=np.nanmax(Tp), zorder=1, levels = 100, linestyles='none')
    cbar = plt.colorbar(contour, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Peak period [s]')
    cbar.set_ticks(np.arange(0, Tp_upper, 1))

def draw_DirVector(ax, x, y, Dir, mag):
   
    Dir = 90.0 - Dir
    Dir[Dir < -180.0] += 360.0

    u = np.sin(np.deg2rad(Dir)) * mag
    v = np.cos(np.deg2rad(Dir)) * mag

    ax.quiver(x[0::2, 0::2],
              y[0::2, 0::2],
              u[0::2, 0::2],
              v[0::2, 0::2],
              zorder=2,
              scale=10, 
              scale_units='inches', headwidth=4, 
              headlength=3, 
              headaxislength=3,
              color='k')


def gif_shoreline_evolution(run, obs, time_obs, idx, path):
    '''Create a gif with the shoreline evolution'''

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'bold'})

    datedict = dict(boxstyle="round",
                    edgecolor="black",
                    fc = "white")    
    run_obs = run[idx]
    images = []

    lower, upper = get_limits(run, obs, [], 10)
    trs = np.linspace(1, len(run_obs[0,:]), len(run_obs[0,:]))

    for i, t in enumerate(time_obs):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(trs, obs[i,:], color='m', linewidth=1.5, label = 'Observed shoreline')
        ax.plot(trs, run_obs[i,:], color='r', linewidth=1.5, label = 'Simulated shoreline', linestyle='--')
        ax.legend(loc = 'upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))
        ax.grid(True)
        ax.set_ylim([lower, upper])
        ax.set_xlim([1, len(run_obs[i,:])+1])
        ax.set_facecolor((0, 0, 0, 0.15))
        despl = [0.13*(np.diff([1, len(run_obs[i,:])+1])), 0.04*(np.diff([lower, upper]))]
        text_in = f'{t.day}/{t.month}/{t.year}'
        ax.text(ax.get_xlim()[1]-despl[0], ax.get_ylim()[1]-despl[1],
            text_in, bbox=datedict, fontweight='bold', verticalalignment='center', horizontalalignment='center', size = 8)
        ax.set_ylabel('Distance from baseline [m]')
        ax.set_xlabel('Transect number')
        filename = f'{path}Shoreline_{i}.png'
        fig.savefig(filename)
        plt.close(fig)
        images.append(imageio.imread(filename))

    imageio.mimsave(f'{path}shoreline_evolution.gif', images, fps=2)

    for filename in os.listdir(path):
        if filename.startswith("Shoreline_") and filename.endswith(".png"):
            os.remove(os.path.join(path, filename))

    return

def draw_generic_var(ax, x, y, var, cmap, alpha, var_id, units):
    cbar_label = f'{var_id} [{units}]'
    
    # Calcular el límite superior e inferior redondeado
    upper = np.ceil(np.nanmax(var))
    lower = np.floor(np.nanmin(var))
    
    # Calcular el intervalo para tener 10 ticks
    tick_interval = (upper - lower) / 10
    ticks = np.arange(lower, upper + tick_interval, tick_interval)
    
    contour = ax.contourf(x, y, var, cmap=cmap, extend='both', alpha=alpha, vmin=lower, vmax=upper, zorder=1, levels=100, linestyles='none')
    cbar = plt.colorbar(contour, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks(ticks)
    
    return contour, cbar

def draw_var_nodes(ax, x, y, faces, var, cmap, alpha, var_id, units):

    # x = ds['mesh2d_node_x'].values
    # y = ds['mesh2d_node_y'].values
    cbar_label = f'{var_id} [{units}]'
    
    upper = np.ceil(np.nanmax(var))
    lower = np.floor(np.nanmin(var))
    
    tick_interval = (upper - lower) / 10
    ticks = np.arange(lower, upper + tick_interval, tick_interval)

    # faces = ds['mesh2d_face_nodes'].values - 1  # 1-based para 0-based

    triangles = []
    for face in faces:
        if not np.any(face == -1):  
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])

    triangles = np.array(triangles)

    triang = tri.Triangulation(x, y, triangles=triangles)

    mask = np.all(~np.isnan(triang.triangles), axis=1)
    triang.set_mask(~mask)

    plt.figure(figsize=(10, 10))
    contour = plt.tricontourf(triang, var, levels=100, cmap=cmap, extend='both', alpha=alpha, vmin=lower, vmax=upper, linestyles='none')
    cbar = plt.colorbar(contour, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks(ticks)

def draw_var_edges(ax, x, y, var, edges, cmap, alpha, var_id, units):
    cbar_label = f'{var_id} [{units}]'
    
    upper = np.ceil(np.nanmax(var))
    lower = np.floor(np.nanmin(var))
    
    tick_interval = (upper - lower) / 10
    ticks = np.arange(lower, upper + tick_interval, tick_interval)

    # Criar triangulação
    triang = tri.Triangulation(x, y)
    
    # Associar valores dos edges aos nós (aqui usamos a média dos valores dos edges conectados a cada nó)
    node_values = np.zeros_like(x)
    node_counts = np.zeros_like(x)
    for i, edge in enumerate(edges):
        node_values[edge] += var[i]
        node_counts[edge] += 1

    node_values = node_values / node_counts

    # Plotar contorno
    contour = ax.tricontourf(triang, node_values, levels=100, cmap=cmap, extend='both', alpha=alpha, vmin=lower, vmax=upper, linestyles='none')
    cbar = plt.colorbar(contour, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks(ticks)
    ax.scatter(x, y, color='red', s=10, edgecolor='k')  # Opcional: para visualizar os pontos

def draw_var_faces(ax, x, y, faces, var, cmap, alpha, var_id, units):
    cbar_label = f'{var_id} [{units}]'
    
    # upper = np.ceil(np.nanmax(var))
    # lower = np.floor(np.nanmin(var))

    upper = np.nanmax(var)
    lower = np.nanmin(var)

    
    tick_interval = (upper - lower) / 10
    ticks = np.arange(lower, upper + tick_interval, tick_interval)

    faces = faces - 1  # Ajustando índice de 1-based para 0-based

    # Verificar se os índices das faces estão dentro do intervalo válido
    max_index = np.max(faces)
    if max_index >= len(x):
        raise ValueError(f"Índice máximo {max_index} fora do intervalo para os pontos x com comprimento {len(x)}.")

    triangles = []
    face_values = []
    for i, face in enumerate(faces):
        if not np.any(face == -1):
            if len(face) == 4:  # Garantir que a face tem 4 vértices
                triangles.append([face[0], face[1], face[2]])
                face_values.append(var[i])
                triangles.append([face[0], face[2], face[3]])
                face_values.append(var[i])
            elif len(face) == 3:  # Caso a face já seja um triângulo
                triangles.append([face[0], face[1], face[2]])
                face_values.append(var[i])

    triangles = np.array(triangles)
    face_values = np.array(face_values)

    triang = tri.Triangulation(x, y, triangles=triangles)

    mask = np.all(~np.isnan(triang.triangles), axis=1)
    triang.set_mask(~mask)

    # Plotar usando tripcolor
    tpc = ax.tripcolor(triang, face_values, cmap=cmap, alpha=alpha, vmin=lower, vmax=upper)
    cbar = plt.colorbar(tpc, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks(ticks)