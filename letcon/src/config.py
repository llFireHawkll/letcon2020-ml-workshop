'''
File: config.py
Project: letcon2020-ml-workshop
File Created: Monday, 17th August 2020 4:24:22 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 2:38:57 am
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
import os 

__all__ = ['LOGGER_OUTPUT_PATH', 'ARTIFACT_OUTPUT_PATH']

### LOGGER CONFIG ###
LOGGER_OUTPUT_PATH = str(os.getcwd().replace("\\",'/')) + '/Logs/'
print('Initializing Logger Path...')

# Check if the above path exists or not
if os.path.exists(LOGGER_OUTPUT_PATH):
    print('Logger Path Exists...')
else:
    print('Creating Logger Path...')
    os.mkdir(LOGGER_OUTPUT_PATH)
    
print('Path for Logger-->{0}'.format(LOGGER_OUTPUT_PATH))


### DATA & MODEL ARTIFACTS CONFIG ###
ARTIFACT_OUTPUT_PATH = str(os.getcwd().replace("\\",'/')) + '/Artifacts/'
print('Initializing Artifacts Path...')

# Check if the above path exists or not
if os.path.exists(ARTIFACT_OUTPUT_PATH):
    print('Artifacts Path Exists...')
else:
    print('Creating Artifacts Path...')
    os.mkdir(ARTIFACT_OUTPUT_PATH)
    
print('Path for Artifacts-->{0}'.format(ARTIFACT_OUTPUT_PATH))