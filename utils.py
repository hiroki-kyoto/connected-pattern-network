# Utilities for STCPNs
import json

def handle_conf_json(conf_json):
    assert type(conf_json) == str or type(conf_json) == unicode

    if type(conf_json) == str:
        conf_json = str(conf_json).decode('utf8')

    conf = json.loads(conf_json, encoding='utf8')
    return conf