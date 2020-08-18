'''
File: logger.py
Project: logger
File Created: Monday, 17th August 2020 4:25:50 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 3:22:45 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from letcon.src.config import LOGGER_OUTPUT_PATH

import sys
import logging

def logger(name=None):
    """[Logger for logging all the details of the program]

    Args:
        name ([type], optional): [description]. Defaults to None.

    Returns:
        [logger]: [description]
    """
    logging.basicConfig(filename=LOGGER_OUTPUT_PATH + 'logs.txt',
                        level=logging.INFO,
                        format="[%(asctime)s] %(name)s: %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )

    logger_function = logging.getLogger(name=name)
    return logger_function