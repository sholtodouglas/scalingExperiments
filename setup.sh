sudo add-apt-repository ppa:longsleep/golang-backports -y
sudo apt update
sudo apt install golang-go
env GOPATH=/root/go
sudo apt-get install graphviz gv
go install github.com/google/pprof@latest
pip install jupyter optax  tensorflow_datasets
pip install dm-haiku==0.0.5