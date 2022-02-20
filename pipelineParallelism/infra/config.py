cluster_config = {
    'nodes': 4,
    'pipeline_length': 2,
    'name': "test",
    "project": "learning-from-play-303306",
    "accelerator_type": "v2-8",
    "zone": "us-central1-f",
    "preemptible": False,
    "redis_password": "5241590000000000" # the default
}

constant_args = f"--zone={cluster_config['zone']} --project={cluster_config['project']}"


# the formatting in this section is weird, but precisely maps to the tmuxinator file.
# so that it is easily modifiable to anyone with familiarity there.
if __name__ == "__main__":

    tmuxinator_header = '''
name: scaling
root: ~/
windows:
- servers:
    layout: tiled
    panes:'''

    with open('.tmuxinator.yaml', 'w') as file:

        
        panes = "".join([f'''
    - test_{i}:
      - clear''' for i in range(0, cluster_config['nodes'])] )

        file.write(tmuxinator_header + panes)


