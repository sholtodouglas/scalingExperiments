cluster_config = {
    'nodes': 6,
    'channels': 2,
    'name': "test",
    "project": "learning-from-play-303306",
    "accelerator_type": "v2-8",
    "zone": "us-central1-f"
}

constant_args = f"--zone={cluster_config['zone']} --project={cluster_config['project']}"
