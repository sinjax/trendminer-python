import logging; 
import logging.config
import os

import warnings


warnings.simplefilter("ignore")
if os.path.exists("logconfig.ini"): 
	logging.config.fileConfig("logconfig.ini")
	logger = logging.getLogger("root")
	# logger.debug("Logger prepared!")