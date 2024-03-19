import logging
from datetime import datetime
'''
In this configuration, all messages of level DEBUG and above are logged to a file named app.log, 
while messages of level INFO and above are also printed to the console. 
Adjust the level and format as needed for your application.
'''
def setup_logging():
    
    # Get the current date and time
    now = datetime.now()
    
    # Format the date and time string with up to minute accuracy
    date_time_str = now.strftime('%Y-%m-%d %H:%M')
    
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{date_time_str}.log',
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

