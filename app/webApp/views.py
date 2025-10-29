from django.shortcuts import render
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from webApp.models import Operations
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
import requests
import os
from pathlib import Path
from urllib.parse import urlparse
import json


class PaperSnitchLoginView(LoginView):

    template_name = "registration/login.html"


class HomePageView(View):
    template_name = "home.html"

    def get(self, request):
        # Prende i tipi di sezione dal database (modificabili da admin)
        operations = Operations.objects.all()
        operation = [ct.name for ct in operations]

        return render(request, self.template_name, {"operations": operation})
