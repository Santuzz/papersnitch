DOMAIN="paper-snitch.ing.unimore.it"
EMAIL="davi9898@gmail.com"

docker compose run --rm certbot certonly --webroot -w /var/www/certbot \
    -d ${DOMAIN} \
    --email ${EMAIL} \
    --agree-tos