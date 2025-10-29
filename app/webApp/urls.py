"""
URL configuration for web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from .views import HomePageView, PaperSnitchLoginView
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("accounts/", include("django.contrib.auth.urls")),
    # path("accounts/login/", views.PaperSnitchLoginView.as_view(), name="login"),
    path("accounts/login/", auth_views.LoginView.as_view(), name="login"),
    path("home/", HomePageView.as_view(), name="homepage"),
    # path("accounts/profile/", views.profile, name="profile"),
]
