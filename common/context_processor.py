from django.templatetags.static import static

def site_data(request):
    data = {
        "name": "cCalcX",
        "title": "Analytical Zip Generator",
        "description": "Maritime analysis from JSON. Analytical data is wrapped in a ZIP file, containing a Matplotlib image, an interactive HTML file, and an operation log.",
        "og_image" : static("android-chrome-512x512.png")
    }

    return data

def hak_common_context(request):
    context = {
        "site_data" : site_data(request)
    }
    return context