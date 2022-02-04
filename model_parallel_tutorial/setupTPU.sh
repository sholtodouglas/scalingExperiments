
gcloud alpha compute tpus tpu-vm create lfp1 --zone=us-central1-f --accelerator-type=v3-8 --version=v2-alpha --project learning-from-play-303306
gcloud alpha compute tpus tpu-vm ssh lfp1 --zone=us-central1-f --project learning-from-play-303306 -- -L 8888:localhost:8888