from datetime import datetime

from python_package.utils.common import get_uuid
from python_package import config
from python_package.utils.logger import define_log_config, get_logger

def hello_world(word):
    log_config = define_log_config(log_dir=config.LOGS_DIR, log_filename_prefix="TEMPLATE", uuid=get_uuid(), creation_time=datetime.utcnow())
    log = get_logger(log_config=log_config, name=__name__)
    
    log.info("Logger defined!")
    log.info("Saying hello...")
    
    return f"Hello, {word}!"


if __name__ == "__main___":
    hello_world()
