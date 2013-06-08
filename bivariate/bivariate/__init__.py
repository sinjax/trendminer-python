import logging; 
import logging.config

logging.config.fileConfig("logconfig.ini")
logger = logging.getLogger("root")
logger.debug("Logger prepared!")