# This file handles the establishment and management of a cluster of TPU-VMS

from .config import cluster_config, constant_args
from .utils import construct_cluster_names, list_tpus
from os import system
import psutil


'''
'''

import yaml 

def make_tmux():
    names = construct_cluster_names(cluster_config['nodes'])

    def make_pane(name):
        pane = {name: [
            "clear"
            ]}
        return pane

    data = {
    "name": "scaling",
    "root": "~/",
    "windows": [
        {"servers": {
            "layout": "tiled",
            "panes": [make_pane(name) for name in names],
        }
        } 
    ],
    }

    file = open(".tmuxinator.yaml", "w")
    yaml.dump(data, file)
    file.close()


def tmux(command):
    system('tmux %s' % command)

def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)

def connect():
    to_connect = construct_cluster_names(cluster_config['nodes'])
    connections = psutil.net_connections()
    ports = [conn.laddr.port for conn in connections]

    existing_tpus = list_tpus()['nodes'] 
    ready_tpus = [node['name'].split('/')[-1] for node in existing_tpus if node['state'] == 'READY']


    tmux("set -g pane-active-border-style bg=default,fg=magenta")
    tmux("set -g pane-border-style fg=green")

    # start the existing, but stopped nodes
    for name in to_connect:

        pane  = int(name.split('_')[-1])
        port = 8800 + pane
        tmux(f'select-pane -t {pane}')

        # if it is the correct type of node
        if name in ready_tpus:
            # tmux("set pane-border-style bg=green,fg=green") # make it red for failing to connect
            if port in ports:
                print(f"{port} already in use - indicates connection exists")
            else:
                
                tmux_shell(f"gcloud alpha compute tpus tpu-vm ssh {name} {constant_args} -- -L {port}:localhost:{port}")
        else:
            pass
            # tmux("set pane-border-style bg=red,fg=red") # make it red for failing to connect

def run(cmd):
    panes_to_connect_to = [int(name.split('_')[-1]) for name in construct_cluster_names(cluster_config['nodes'])]
    connections = [conn.laddr.port for conn in psutil.net_connections()] 
    active_panes = [pane for pane in panes_to_connect_to if 8800 + pane in connections]

    for pane in active_panes:
        tmux(f'select-pane -t {pane}')
        tmux_shell(cmd)


def clear_all():
    panes_to_connect_to = [int(name.split('_')[-1]) for name in construct_cluster_names(cluster_config['nodes'])]
    for pane in panes_to_connect_to:
        tmux(f'select-pane -t {pane}')
        tmux_shell('clear')