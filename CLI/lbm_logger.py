import logging
import os

def setup_lbm_logger(name="lbm_debug", log_file="lbm_debug.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger

lbm_debug_logger = setup_lbm_logger()
