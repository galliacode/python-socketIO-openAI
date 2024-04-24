#!/bin/bash

cd /home/ec2-user/analitiq
sudo docker-compose down
git pull
git fetch origin
git reset --hard dev
bash github-action-script/get_env.sh
sudo docker-compose up -d
