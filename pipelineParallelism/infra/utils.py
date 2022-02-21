# This file handles the establishment and management of a cluster of TPU-VMS
import functools
import os
import requests
import subprocess
from tqdm import tqdm


from fabric import Connection
import os

from infra.config import cluster_config, constant_args

# @functools.lru_cache() # TODO this can error if it has been a while since it is called. 
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()

def check_tpu(name):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes/{name}",
        headers=headers)

    return response.json()


def list_tpus():
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes",
        headers=headers)

    return response.json()


def create_tpu(name):
    print(f"Creating {name}")
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
        'Content-Type': 'application/json',
    }

    params = (
        ('node_id', name),
    )

    data = {"accelerator_type":
                cluster_config['accelerator_type'],
            "runtime_version":
                'v2-alpha',
            "network_config":
                {"enable_external_ips": True},
            "schedulingConfig": 
                {"preemptible": cluster_config['preemptible']},
            }

    response = requests.post(f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes",
                             headers=headers, params=params, json=data)

    print(response.json())

    return response.status_code == 200


def start_tpu(name):
    print(f"Starting {name}")
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.post(
        f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes/{name}:start",
        headers=headers)

    return response.json()


def stop_tpu(name):
    print(f"Stopping {name}")
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.delete(
        f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes/{name}:stop",
        headers=headers)

    return response.json()


def delete_tpu(name):
    print(f"Deleting {name}")
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.delete(
        f"https://tpu.googleapis.com/v2alpha1/projects/{cluster_config['project']}/locations/{cluster_config['zone']}/nodes/{name}",
        headers=headers)

    return response.json()

def construct_cluster_names(N: int):
    return [f"{cluster_config['name']}-{n}" for n in range(0, N)]

def scale_cluster():
    existing_tpus = list_tpus().get('nodes', []) 

    # determine what we need to create 
    to_construct = construct_cluster_names(cluster_config['nodes'])

    # start the existing, but stopped nodes
    for node in existing_tpus:
        name = node['name'].split('/')[-1]
        # if it is the correct type of node
        if name in to_construct  and node['acceleratorType'] == cluster_config['accelerator_type']:
            # remove it from what we need to construct
            to_construct.remove(name)
            # start any that are stopped
            if node['state'] == 'STOPPED':
                start_tpu(name)
            if node['state'] == 'PREEMPTED':
                delete_tpu(name)
                start_tpu(name)

    # create the remainder
    for remaining in to_construct:
        res = create_tpu(remaining) # TODO: if res failed. 

    

def validate_cluster():
    print('Validating cluster creation')
    for name in tqdm(construct_cluster_names(cluster_config['nodes'])):
        try:
            if check_tpu(name)['state'] != 'READY':
                print(f"Failed: {name}")
        except:
            print(f"TPU {name} not found")


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def get_pipelines():
    ''' 
    Takes the currently active nodes and arranges them into full length pipelines
    If there are insufficient nodes for the final pipeline, they are ignored. 
    '''
    tpus = list_tpus().get('nodes', []) 

    if len(tpus) < cluster_config['pipeline_length']:
        raise Exception('Insufficient Devices to form a pipeline')

    # create a connection object for each tpu to allow for easy file copy
    for tpu in tpus:
        tpu['connection_object'] = Connection(tpu['networkEndpoints'][0]['accessConfig']['externalIp'], connect_kwargs={
                                      "key_filename": os.path.expanduser('~/.ssh/google_compute_engine'), })

    # arrange them in pipelines
    complete_pipelines = [p for p in chunks(tpus, cluster_config['pipeline_length']) if len(p) == cluster_config['pipeline_length']]
    
    return complete_pipelines


