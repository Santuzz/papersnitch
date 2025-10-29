docker compose exec django-web python manage.py makemigrations $@
docker compose exec django-web python manage.py migrate