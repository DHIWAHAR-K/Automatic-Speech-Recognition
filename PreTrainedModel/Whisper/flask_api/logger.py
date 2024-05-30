# logger.py
import os
import logging

# Check if the "logs" directory exists, if not, create it
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Set up server logging
server_logger = logging.getLogger('server')
server_logger.setLevel(logging.INFO)
server_handler = logging.FileHandler(os.path.join(logs_dir, 'requests.log'))
server_formatter = logging.Formatter('%(asctime)s - INFO - %(message)s')
server_handler.setFormatter(server_formatter)
server_logger.addHandler(server_handler)