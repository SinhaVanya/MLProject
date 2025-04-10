import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #log file name with timestamp
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #log file path
os.makedirs(log_path,exist_ok=True) #create directory if it doesn't exist

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE) #full path to the log file

logging.basicConfig(
    filename=LOG_FILE_PATH, #log file path
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s', #log message format
    level=logging.INFO #log level
)

if __name__=="__main__":
    logging.info("Logging has started") #log message when the script is run directly