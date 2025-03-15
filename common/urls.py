
from django.urls import path
from .views import *
from django.views.generic import TemplateView
from django.contrib.sitemaps.views import sitemap
from .sitemaps import StaticSitemap

app_name = "common"

sitemap_list = {
    'static': StaticSitemap
}


urlpatterns = [
    path("", landing_view, name="common_landing"),
    path("example.json/", example_json, name="example_json"),
    path("sitemap.xml/", sitemap, {'sitemaps': sitemap_list}, name='django.contrib.sitemaps.views.sitemap'),
    path('robots.txt', TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),

    
]