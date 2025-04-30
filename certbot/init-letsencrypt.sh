#!/bin/bash

domains=(ml-api.socialsight.me)
email="muhammadumairkhan945@gmail.com" 
rsa_key_size=4096
data_path="./certbot"
staging=0 # Set to 1 if testing

mkdir -p "$data_path/www"

docker-compose run --rm \
  -v "$PWD/certbot/www:/var/www/certbot" \
  -v "$PWD/certbot/etc:/etc/letsencrypt" \
  -v "$PWD/certbot/lib:/var/lib/letsencrypt" \
  certbot/certbot certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  --email $email \
  --agree-tos \
  --no-eff-email \
  -d ${domains[@]} \
  ${staging:+--staging}
