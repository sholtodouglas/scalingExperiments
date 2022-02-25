# Step 0

Create a config at infra/config.py


```
# generate the tmuxinator pane
python infra/config.py

# run tmuxinator to set up the work space
tmuxinator start scaling -p infra/.tmuxinator.yaml

```


# Step 1

Create a cluster, which will have 1 leader (coordinating node), and N worker nodes. 


```
python 
```
