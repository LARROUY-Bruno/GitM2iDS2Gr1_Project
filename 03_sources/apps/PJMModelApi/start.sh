#!/bin/bash
app="pjmmodelapi"
docker build -t ${app} .
docker run -d --network docker-cluster_prj_network -p 5002:5002 \
  --name=${app} \
  -v $PWD:/app ${app}
