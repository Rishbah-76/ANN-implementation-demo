import yaml
import time 

def read_config(filePath):
    with open(filePath) as config_file:
        content = yaml.safe_load(config_file)
    return content

def get_timestamp(filename):
    timestamp=time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{filename}_at_{timestamp}"
    return unique_name
