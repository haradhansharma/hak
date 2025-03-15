from django.contrib import sitemaps
from django.utils import timezone
from django.urls import reverse


class StaticSitemap(sitemaps.Sitemap):
    priority = 0.5
    changefreq = 'monthly'
    
    def items(self):
        return [
            'common:common_landing',
            'common:example_json' 
            
        ]
        
    def lastmod(self, obj):
        return timezone.now()
    
    def location(self, item):
        return reverse(item)
    