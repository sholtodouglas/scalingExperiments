To achieve pipeline parallelism - we'll need to coordinate a whole cluster of machines. The optimal hardware would be something like a TPU pod, which is made up of up to 32 TPU boards (each with 8 TPU cores), connected with a high speed interconnect.  I don't have access to one, but I do have access to a large cluster of separate TPUv2 boards - which is arguably more representative of the clusters of A100s (or similar) that are in use outside GCP. That makes them an excellent playground to explore inter-machine parallelism!

As with the previous post, this is focused tightly on being educational and looking at some of the design decisions which might have gone into writing a framework like Deepspeed. 

- 

# Setup

As we'll be using a whole cluster of machines - we'll need a way to orchestrate them. Normally, I'd opt for kubernetes through Google Kubernetes Engine (GKE) - but it appears to only support the older TPU nodes which require a separate host CPU driving them (and are commensurately much slower). Instead, we'll write our own simple orchestration functions (I'm quite confident TPU-VM's will be available through GKE in time).

# Distributed Systems

Two libaries stand out

- Ray: A distributed systems library for python. It sets up a single head + multiple worker nodes, but theoretically the head shouldn't be a bottleneck as it does not copy data to the head in order to transfer it between worker nodes.
- mpi4jax: Zero copy, multi-host communication of JAX arrays. No orchestrating node is required - all nodes send and receive directly to eachtother. 

mpi4jax is a lower level library and is likely to introduce more complexity into the code -  but it may be faster as it has been directly optimised for the transfer of jax arrays from GPU/TPU memory.  Lets test! Regardless, we'll still use Ray for multiprocessing of orchestration operations from our local machine. 



# Design

Pre-emption seems reasonably common, so lets make it fault tolerant from the start. 

We should have a continously running process which performs the following steps

 - Tries to create the desired config
 - - Checks the currently active nodes, and constructs the best possible configuration from it. 
 - - Sets that running
 - - If one of the existing ones is pre-empted, cancel the program somehow, recompute optimal arrangement
 - - If a new one is added, recompute optimal config. Wait till next save epoch to update



 # Setup 

  Working with and debugging a distributed environment is a little more annoying! While we're getting started, I set up a tmuxinator that sshs into everything. 

  To ensure the tmux looks pretty, copy the following into ~/.tmux.conf

  

  tmuxinator start scaling -p .tmuxinator.yaml



  # GCP setup

  Create a project with TPU accces

  Create an ssh key called 'google_compute_engine', and add it to the project.  
  '''
ssh-keygen -t rsa


gcloud compute os-login ssh-keys add \
    --key-file=KEY_FILE_PATH.pub \
    --project=PROJECT

  '''

 


gcloud compute os-login ssh-keys add \
    --key-file= /home/sholto/.ssh/google_compute.pub \
    --project=learning-from-play-303306



Enable each machine to access the others
https://www.open-mpi.org/faq/?category=running#missing-prereqs
https://github.com/NAThompson/mpi_clustering


ssh-keygen -t rsa -f $HOME/.ssh/id_rsa -N '' -C "MPI Keys"

Need to make sure each device shares an ssh key