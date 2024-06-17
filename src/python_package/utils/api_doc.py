from flask_restx import fields


def set_swagger_fields(api):

    fields_continuous_release = api.model(
        "continuous release",
        {
            "release_duration_h": fields.Float(
                required=True,
                description="duration of the continuous release (hours) (0: deatcivated)",
                example=0.0,
                min=0,
                max=120,
            ),
            "release_dt_s": fields.Integer(
                required=True,
                description="time step between release pulses (seconds) (0: deatcivated)",
                example=0,
                min=0,
                max=86400,
            ),
        },
    )

    fields_spill_point = api.model(
        "spill point",
        {
            "substanceName": fields.String(
                required=True,
                description="substance name from database",
                example="Tia Juana",
            ),
            "longitude": fields.Float(
                required=True,
                description="longitude coordinate of the spill point (º)",
                min=-180,
                max=180,
                example=49.5,
            ),
            "latitude": fields.Float(
                required=True,
                description="longitude coordinate of the spill point (º)",
                min=-90,
                max=90,
                example=27.6,
            ),
            "initial_thickness": fields.Float(
                required=True,
                description="initial thickness of the oil slick (m)",
                example=0.02,
            ),
            "minimum_thickness": fields.Float(
                required=True,
                description="minimum thickness of the oil slick (m)",
                example=0.0001,
            ),
            "initial_width_lon": fields.Float(
                required=True,
                description="initial lenght along the longitude axis of the oil slick (m)",
                example=50.5,
            ),
            "initial_width_lat": fields.Float(
                required=True,
                description="initial lenght along the latitude axis of the oil slick (m)",
                example=50.5,
            ),
            "mass": fields.Float(
                required=True,
                description="total mass released (kg)",
                example=300000,
            ),
            "releaseTime": fields.String(
                required=True,
                description="Initial UTC date and time",
                example="2021-04-15T12:00:00Z",
                pattern="\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ",
            ),
        },
    )

    vel2d_variable_names = api.model(
        "varaible names - velocity 2d",
        {
            "u": fields.String(
                required=False,
                description="name of the 'u' variable in the netCDF file",
                example="u",
            ),
            "v": fields.String(
                required=False,
                description="name of the 'v' variable in the netCDF file",
                example="v",
            ),
        },
        skip_none=True,
        allow_null=True,
    )

    vel2d_coordinate_names = api.model(
        "coordinate names - velocity 2d ",
        {
            "longitude": fields.String(
                required=False,
                description="name of the 'longitude' coordinate in the netCDF file",
                example="lon",
            ),
            "latitude": fields.String(
                required=False,
                description="name of the 'latitude' coordinate in the netCDF file",
                example="lat",
            ),
            "time": fields.String(
                required=False,
                description="name of the 'time' coordinate in the netCDF file",
                example="time",
            ),
        },
        skip_none=True,
        allow_null=True,
    )

    fields_currents_provider_name = api.model(
        "currents provider name",
        {
            "downloader": fields.String(
                required=False,
                description="name of the dowloader implementation",
                example="datahubclient",
            ),
            "providerName": fields.String(
                required=False,
                description="name of the currents provider (opendap-name)",
                example="cmems_global_currents",
            ),
            "variableNames": fields.Nested(
                vel2d_variable_names, skip_none=True, allow_null=True
            ),
            "coordinateNames": fields.Nested(
                vel2d_coordinate_names, skip_none=True, allow_null=True
            ),
            "ncPath": fields.List(
                fields.String(
                    required=False,
                    description="Currents netcdf path",
                    example="tests/utils/data/currents.nc",
                )
            ),
        },
        skip_none=True,
        allow_null=True,
    )

    fields_winds_provider_name = api.model(
        "winds provider name",
        {
            "downloader": fields.String(
                required=False,
                description="name of the dowloader implementation",
                example="datahubclient",
            ),
            "providerName": fields.String(
                required=False,
                description="name of the winds provider (opendap-name)",
                enum=["noaa_gfs_winds", "unige"],
                example="noaa_gfs_winds",
            ),
            "variableNames": fields.Nested(vel2d_variable_names),
            "coordinateNames": fields.Nested(vel2d_coordinate_names),
            "ncPath": fields.List(
                fields.String(
                    required=False,
                    description="Winds netcdf path",
                    example="tests/utils/data/winds.nc",
                )
            ),
        },
    )

    # fields_currents_depthavg_provider_name = api.model(
    #     "currents depthavg provider name",
    #     {
    #         "providerName": fields.Raw(
    #             required=False,
    #             description="name of the currents depth avergae provider (opendap-name)",
    #             enum=["cmems_global_depthavg"],
    #             example=None,
    #         ),
    #     },
    # )

    # fields_waves_provider_name = api.model(
    #     "waves provider name",
    #     {
    #         "providerName": fields.Raw(
    #             required=False,
    #             description="name of the currents provider (opendap-name)",
    #             example=None,
    #         ),
    #     },
    # )

    fields_forcings = api.model(
        "forcings",
        {
            "currents": fields.Nested(
                fields_currents_provider_name, skip_none=True, allow_null=True
            ),
            "winds": fields.Nested(
                fields_winds_provider_name, skip_none=True, allow_null=True
            ),
            "waves": fields.Raw(description="waves provider dictionary"),
            "currents_depthavg": fields.Raw(
                description="depth-average currents provider dictionary"
            ),
        },
    )

    fields_environment_variables = api.model(
        "environment variables",
        {
            "seaTemperature": fields.Float(
                required=True,
                description="Sea temperature (ºC)",
                example=20,
            ),
            "seaDensity": fields.Float(
                required=True,
                description="Sea density (kg/m3)",
                example=1025,
            ),
            "airTemperature": fields.Float(
                required=True,
                description="Air temperature (ºC)",
                example=20,
            ),
        },
    )

    fields_calibration_coefficients = api.model(
        "calibration coefficients",
        {
            "currentsFactor": fields.Float(
                required=True,
                description="currents calibration factor",
                example=1,
            ),
            "windsDragAlfa": fields.Float(
                required=True,
                description="wind drag coefficient (alfa)",
                example=0.03,
            ),
            "windsDragBeta": fields.Float(
                required=True,
                description="wind drag coefficient (beta)",
                example=0,
            ),
            "wavesFactor": fields.Float(
                required=True,
                description="waves calibration factor",
                example=0.15,
            ),
            "particleDispersionFlag": fields.Integer(
                required=True,
                description="particle dispersion activate-flag (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "horizontalDispersion": fields.Float(
                required=True,
                description="horizontal dispersion coefficient",
                example=50,
            ),
            "verticalDispersion": fields.Float(
                required=True,
                description="vertical dispersion coefficient",
                example=0.01,
            ),
            "SPdteSupfAgua": fields.Float(
                required=True,
                description="SPdteSupfAgua",
                example=1,
            ),
        },
    )

    fields_model_setup = api.model(
        "modelSetup",
        {
            "simulationMode": fields.Integer(
                required=True,
                description="simulation mode flag (1:2D, 2:3D)",
                example=1,
                min=1,
                max=2,
                multiple=1,
            ),
            "forwardBackward": fields.Integer(
                required=True,
                description="forwards/backwards flag (1:forwards, 2:backwards)",
                example=1,
                min=1,
                max=2,
                multiple=1,
            ),
            "nParticles": fields.Integer(
                required=True,
                description="number of particles used in each release pulse",
                example=1000,
                min=10,
                max=10000,
                multiple=1,
            ),
            "dtSec": fields.Integer(
                required=True,
                description="time step of the model (seconds)",
                example=60,
                min=1,
                max=21600,
                multiple=1,
            ),
            "firstTimeSaved": fields.Integer(
                required=True,
                description="first time saved since forcing initialization (hours)",
                example=0,
                min=0,
                multiple=1,
            ),
            "saveParticles": fields.Integer(
                required=True,
                description="save particles results (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "saveGrid": fields.Integer(
                required=True,
                description="save concentrations results (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "saveProperties": fields.Integer(
                required=True,
                description="save spill properties results (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
        },
    )

    fields_cfg = api.model(
        "cfg",
        {
            "gridName": fields.String(
                required=True,
                description="filename of the model grid",
                example="grid.dat",
            ),
            "coastlineAdherenceFlag": fields.Integer(
                required=True,
                description="adherence based on special index (0:deactivated, 1:activated)",
                example=0,
            ),
            "seaKinematicVisc": fields.Float(
                required=True,
                description="sea kinematic viscosity",
                example=1e-06,
            ),
            "spreadingFlag": fields.Integer(
                required=True,
                description="simulate spreading (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "spreadingType": fields.Integer(
                required=True,
                description="spreding algorithm (1:Equivalent dif.[ADIOS2], 2:Ellipsoidal[Lehr], 3:Rabeh-Kolluru[MOHID])",
                example=1,
                min=1,
                max=3,
                multiple=1,
            ),
            "spreadingFuelHours": fields.Float(
                required=True,
                description="hours before deactivation of spreading for Fuel simulations",
                example=3,
                exclusiveMin=0,
            ),
            "evaporationFlag": fields.Integer(
                required=True,
                description="simulate evaporation (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "emulsificationFlag": fields.Integer(
                required=True,
                description="simulate emulsification (0:deactivated, 1:activated)",
                example=1,
                min=0,
                max=1,
                multiple=1,
            ),
            "verticalDispersionFlag": fields.Integer(
                required=True,
                description="simulate vertical dispersion (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "dissolutionFlag": fields.Integer(
                required=True,
                description="simulate dissolution (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "volatilizationFlag": fields.Integer(
                required=True,
                description="simulate volatilization (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "sedimentationFlag": fields.Integer(
                required=True,
                description="simulate sedimentation (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "biodegradationFlag": fields.Integer(
                required=True,
                description="simulate biodegradation (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "inputDir": fields.String(
                required=True,
                description="path to inputs folder",
                example="./input/",
            ),
        },
    )

    fields_run = api.model(
        "run",
        {
            "environment": fields.Integer(
                required=True,
                description="select environment (1:sea-water, 2:fresh-water)",
                example=1,
                min=1,
                max=2,
                multiple=1,
            ),
            "blowoutFlag": fields.Integer(
                required=True,
                description="read near-field input (0:deactivated, 1:activated)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "inputDirectory": fields.Integer(
                required=True,
                description="select input directory (1:current working directory, 2:defined in cfg-file)",
                example=2,
                min=1,
                max=2,
                multiple=1,
            ),
            "coastsFlag": fields.Integer(
                required=True,
                description="coastline information (1:only grid-file, 2:grid-file and coast-file)",
                example=2,
                min=1,
                max=2,
                multiple=1,
            ),
            "beachingAlgorithm": fields.Integer(
                required=True,
                description="coastline search algorithm (1:low resolution, 2:high resolution [poligons needed])",
                example=1,
                min=1,
                max=2,
                multiple=1,
            ),
            "restart": fields.Integer(
                required=True,
                description="read initial particles location (0:activate, 1:deactivate)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "interpTimeWinds": fields.Integer(
                required=True,
                description="make time interpolation (0:activate, 1:deactivate)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "interpTimeWaves": fields.Integer(
                required=True,
                description="make time interpolation (0:activate, 1:deactivate)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "interpTimeCurrents": fields.Integer(
                required=True,
                description="make time interpolation (0:activate, 1:deactivate)",
                example=0,
                min=0,
                max=1,
                multiple=1,
            ),
            "executionScheme": fields.Integer(
                required=True,
                description="select execution scheme (1:Euler, 2:Runge-Kutta)",
                example=1,
                min=1,
                max=2,
                multiple=1,
            ),
            "particlesSaveDt": fields.Integer(
                required=True,
                description="time-step in seconds to save particles",
                example=3600,
            ),
            "gridSaveDt": fields.Integer(
                required=True,
                description="time-step in seconds to save grids",
                example=3600,
            ),
            "propertiesSaveDt": fields.Integer(
                required=True,
                description="time-step in seconds to save properties",
                example=3600,
            ),
        },
    )

    fields_other_parameters = api.model(
        "other parameters",
        {"cfg": fields.Nested(fields_cfg), "run": fields.Nested(fields_run)},
    )

    fields_input_json = api.model(
        "input json",
        {
            "project": fields.String(
                required=True,
                description="project name to be used in log filename",
                enum=[
                    "QUALITAS",
                    "BEREADY",
                    "SICMA",
                    "ATHENEA",
                ],
                example="QUALITAS",
            ),
            "domain": fields.String(
                required=True,
                description="teseo model domain (grid, coastlines, polygons)",
                enum=[
                    "qualitas_local",
                    "qualitas_regional",
                    "qualitas_global",
                    "be_ready_spezia",
                ],
                example="qualitas_global",
            ),
            "alias": fields.String(
                description="simulation alias or name",
                example="test simulation",
                required=False,
            ),
            "description": fields.String(
                description="brief description",
                example="test description",
                required=False,
            ),
            "initialTime": fields.String(
                required=True,
                description="Initial UTC date and time of the spill release",
                example="2021-04-15T12:00:00Z",
                pattern="\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ",
            ),
            "duration": fields.Float(
                required=True,
                description="duration of TESEO simulation (hours)",
                example=3.5,
                min=2,
                max=240,
            ),
            "continuous_release": fields.Nested(fields_continuous_release),
            "spillPoints": fields.List(fields.Nested(fields_spill_point)),
            "forcings": fields.Nested(fields_forcings),
            "environmentVariables": fields.Nested(fields_environment_variables),
            "calibrationCoefficients": fields.Nested(fields_calibration_coefficients),
            "modelSetup": fields.Nested(fields_model_setup),
            "otherParameters": fields.Nested(fields_other_parameters),
        },
    )

    return api, fields_input_json
