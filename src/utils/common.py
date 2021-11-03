import yaml

def read_config(filePath):
    with open(filePath) as config_file:
        content = yaml.safe_load(config_file)
    return content