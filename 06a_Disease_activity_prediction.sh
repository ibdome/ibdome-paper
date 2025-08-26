#!/bin/bash
### 5FCV task on Berlin
bash ./external_tools/marugoto/run_crossval_berlin.sh riley
bash ./external_tools/marugoto/run_crossval_berlin.sh cortina
### Deployment on Erlangen
bash ./external_tools/marugoto/run_deploy_erlangen.sh riley
bash ./external_tools/marugoto/run_deploy_erlangen.sh cortina