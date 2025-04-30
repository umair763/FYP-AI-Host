#!/bin/bash

domains=(ml-api.socialsight.me)
email="muhammadumairkhan945@gmail.com"
rsa_key_size=4096
data_path="./certbot"
staging=0 # change to 1 for testing certbot

docker-compose run --rm --entrypoint "\
  certbot certonly --webroot \
  --webroot-path=/var/lib/letsencrypt \
  --email $email \
  --agree-tos \
  --no-eff-email \
  ${staging:+--staging} \
  -d ${domains[*]}" certbot
