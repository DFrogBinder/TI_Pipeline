import logging
'''
In this configuration, all messages of level DEBUG and above are logged to a file named app.log, 
while messages of level INFO and above are also printed to the console. 
Adjust the level and format as needed for your application.
'''
def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='app.log',
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

