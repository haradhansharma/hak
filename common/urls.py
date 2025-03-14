
from django.urls import path
from .views import *


app_name = "common"
urlpatterns = [
    path("", landing_view, name="common_landing"),
    path("example.json/", example_json, name="example_json")
    
]