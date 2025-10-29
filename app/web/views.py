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
