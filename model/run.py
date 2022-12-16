import os
import sys
import json
from datetime import datetime
from model import Controller

def main():
    with open(sys.argv[1], 'r') as f:
        data = f.read()
    obj = json.loads(data)
    timestamp = datetime.today().strftime("%Y%m%d_%H%M")
    title = obj['title']
    experiment_name = f"{title}_{timestamp}"
    output_path = obj['output_path']
    path_to_db = f"{output_path}/{experiment_name}"
    os.mkdir(path_to_db)
    controller = Controller(
        experiment_name, 
        path_to_db
    )
    controller.run_mp(
        obj['num_networks'],
        obj['num_iterations'],
        obj['max_steps'],
        network_params_dict=obj['network_params'],
        world_params_dict=obj['world_params']
    )

if __name__ == "__main__":
    main()
