import pathlib
import json

d = json.loads(pathlib.Path("./src/tha4_api/test_configuration.json").read_text())

for i in d['states']:
    for j in d['states'][i]:
        for k in j:
            k['duration'] -= 0.3
        
pathlib.Path("./src/tha4_api/test_configuration.json").write_text(json.dumps(d, indent=4))