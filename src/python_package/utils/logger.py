# Module to manage loggers, needs some variables to be declared in config.py
# these variables are updated each run directly in main.py during the execution:
#   - creation_time
#   - uuid
#   - log_dir
#   - log_conifg
#
# All of these variables have to be declared in config.py

import os
import logging
from datetime import datetime
from logging import Logger
from pathlib import Path


def define_log_config(
    log_dir_str: str,
    log_filename_prefix: str,
    uuid: str,
    creation_time: datetime,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> dict():
    """Create dictionary with custom handlers, format, uuid and log_path

    Args:
        log_dir (str, log_filename_prefix, optional): [description]. Defaults to logging.INFO, file_level: int = logging.DEBUG, )->dict(.

    Returns:
        [type]: [description]
    """
    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log_filename
    log_filename = (
        f"{log_filename_prefix}_{creation_time.strftime('%Y%m%dT%H%M%SZ')}_{uuid}.log"
    )

    # Create handlers, format and save important content in log_config
    log_config = dict()
    log_config["uuid"] = uuid
    log_config["path"] = Path(log_dir, log_filename).as_posix()
    log_config["log_format"] = logging.Formatter(
        fmt="%(asctime)s > [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_config["console_log"] = logging.StreamHandler()
    log_config["console_log"].setFormatter(log_config["log_format"])
    log_config["console_log"].setLevel(console_level)

    log_config["file_log"] = logging.FileHandler(log_config["path"])
    log_config["file_log"].setFormatter(log_config["log_format"])
    log_config["file_log"].setLevel(file_level)

    return log_config


def get_logger(log_config: dict, name: str, use_uuid: bool = False) -> Logger:
    """Get logger with custom configuration

    Args:
        log_config (dict): dictionary with custom configuration, conatins: handlers, format, uuid and path
        name (str): name of the logger, usually __name__
        use_uuid (bool, optional): To use log_config["uuid"] as name and name argument as child name. Defaults to False.

    Returns:
        Logger: [description]
    """
    logging.root.manager.loggerDict.clear()
    logging.root.handlers.clear()

    if use_uuid:
        log = logging.getLogger(log_config["uuid"]).getChild(name)
    else:
        log = logging.getLogger(name)

    log.addHandler(log_config["file_log"])
    log.addHandler(log_config["console_log"])
    log.setLevel(logging.DEBUG)

    return log
