from builtins import breakpoint
import yaml
option = {}
with open('config_gpu.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
option['server_gpu_id'] = config['server']
process_gpu_id = {}
breakpoint()
for k in config['process'].keys():
        for value in config['process'][k]:
            process_gpu_id[value] = int(k)
option['process_gpu_id'] = process_gpu_id
breakpoint()