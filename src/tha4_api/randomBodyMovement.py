import pathlib
import json
import random

d = json.loads(pathlib.Path(
    "./src/tha4_api/test_configuration.json").read_text())

for i in d['states']:
    for j in d['states'][i]:
        if 'body_rotation' not in [action['action'] for action in j]:
            j.append({
                "action": "body_rotation",
                "y": [0.5, random.uniform(0.3, 0.7)],
                "z": [0.5, random.uniform(0.3, 0.7)],
                "restore": "rapid",
                "transition": "sine",
                "duration": random.uniform(0.4, 0.6)
            })
        elif 'head_rotation' not in [action['action'] for action in j]:
            j.append({
                "action": "head_rotation",
                "x": [0.5, random.uniform(0.3, 0.7)],
                "y": [0.5, random.uniform(0.3, 0.7)],
                "z": [0.5, random.uniform(0.3, 0.7)],
                "restore": "rapid",
                "transition": "sine",
                "duration": random.uniform(0.4, 0.6)
            })

pathlib.Path(
    "./src/tha4_api/test_configuration_edited.json").write_text(json.dumps(d, indent=2))
