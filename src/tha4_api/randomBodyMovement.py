import pathlib
import json
import random

d = json.loads(pathlib.Path(
    "./src/tha4_api/test_configuration.json").read_text())

for i in d['states']:
    for j in d['states'][i]:
        if 'body_rotation' not in [action['action'] for action in j]:
            dur = random.uniform(0.4, 0.6)
            y = [0.5, random.uniform(0.4, 0.6)]
            z = [0.5, random.uniform(0.4, 0.6)]
            
            j.append({
                "action": "body_rotation",
                "y": y,
                "z": z,
                "restore": "rapid",
                "transition": "sine",
                "duration": dur
            })
            
            if random.randint(1, 3) == 2:
                j.append({
                    "action": "body_rotation",
                    "y": y,
                    "z": z,
                    "restore": "rapid",
                    "transition": "sine",
                    "duration": dur
                })
                
        elif 'head_rotation' not in [action['action'] for action in j]:
            j.append({
                "action": "head_rotation",
                "x": [0.5, random.uniform(0.4, 0.6)],
                "y": [0.5, random.uniform(0.4, 0.6)],
                "z": [0.5, random.uniform(0.4, 0.6)],
                "restore": "rapid",
                "transition": "sine",
                "duration": random.uniform(0.4, 0.6)
            })

pathlib.Path(
    "./src/tha4_api/test_configuration_edited.json").write_text(json.dumps(d, indent=2))
