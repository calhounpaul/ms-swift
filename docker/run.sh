DOCKER_DIR_PATH=$(dirname $(realpath $0))
MS_SWIFT_DIR_PATH=$(dirname $DOCKER_DIR_PATH)
CACHE_PATH=~/.cache
WORKDIR=/ms-swift/docker/workdir

docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
    -v $CACHE_PATH:/root/.cache/ \
    -v $MS_SWIFT_DIR_PATH:/ms-swift \
    -w $WORKDIR \
    -it ms-swift-ft