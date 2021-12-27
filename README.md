# Scaling experiments

A minimal implementation of a multi-device sharded transformer training, and a walk through of each component. The intention is educational - we'll build the required elements from the ground up and understand exactly where each computation is going. This is made exceptionally easy by jax, which is beautiful to work with on hardware meshes. 

![Alt Text](https://github.com/sholtodouglas/scalingExperiments/raw/main/media/sharding.gif)

For production ready code, look at the Deepspeed library (for Pytorch), or GPT-J (Jax). Currently this repo focused purely on exploring memory/compute distribution in multi-gpu training - it could be further optimised through using float16, gcp streaming of tfrecords for dataloader, learning rate schedules etc

This code uses the megatron-LM/GPT-J data+tensor parallelism scheme, which is simple and efficient on hardware meshes like TPUs. Soon, I'd like to look at pipeline parallelism, implement ZeRO style sharding - and use Ray to coordinate a K8s cluster of TPUv2s (for all those times you don't have a TPUvX-256!)

This should be run on a TPU (either through GCP / TRC or Colab) as that gives us 8 devices to experiment with. In general, TPUs make training large models much easier - as your needs scale you can use bigger and bigger TPU pods, so its easy to see why Tesla is making their own extensible hardware mesh in Dojo. 

A couple of resources that I've leant on:

- [Lilian Weng's notes on training large models](https://lilianweng.github.io/lil-log/2021/09/24/train-large-neural-networks.html)
- [Ben Wang's GPT-J](https://github.com/kingoflolz/mesh-transformer-jax)
- [Karpathy's MinGPT](https://github.com/karpathy/minGPT)
- 3Blue1Brown's Manim library (to make the gif!)



## Usage
1. Run setup.sh (required for jax memory profiling).
2. Either work through the tutorial notebook, or run train.py. ~1 hour of training produces results identifiably shakespearean/ structured like a play. 


