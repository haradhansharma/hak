import json
import shutil
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from common.C_calc_script import Analysis
from common.forms import GeoJsonUploadForm
from django.template.loader import render_to_string
import tempfile
import os

import logging
log =  logging.getLogger('log')

def example_json(request):
    """Returns a JSON response from a pre-rendered JSON template.

    Args:
        request (HttpRequest): The incoming HTTP request object.

    Returns:
        JsonResponse: A JSON response containing the parsed data from the template.
    """
    json_content = render_to_string("common/example.json")
    data = json.loads(json_content)
    return JsonResponse(data, safe=False)


def landing_view(request):
    
    """Handles file uploads, processes geojson files, and generates a downloadable ZIP archive.

    This view manages file uploads, processes the files using the `Analysis` class,
    and provides a ZIP file for download if processing is successful.

    Args:
        request (HttpRequest): The incoming HTTP request object.

    Returns:
        HttpResponse: A rendered HTML response containing the form and download link (if available).
    """
    
    context = {}
    
    session_id = f"user_{request.user.pk}"
    media_path = settings.MEDIA_ROOT
    zip_sufix = "output_dir"
    format = "zip"
    folder_name = f"{zip_sufix}_{session_id}"
    archive_path = f"{os.path.join(media_path, folder_name)}"
    
    form = GeoJsonUploadForm()
    context["form"] = form
    analysis = Analysis(session_id=session_id, media_path=media_path)
    
    # If a previously generated ZIP exists, return it with a GET request
    if os.path.exists(f"{archive_path}.{format}"):
        path_to_download = f"{settings.MEDIA_URL}{folder_name}.{format}" 
        context["download_path"] = path_to_download   
    
    if request.method == 'POST':   
        form = GeoJsonUploadForm(request.POST, request.FILES, request=request)
        if form.is_valid():
            uploaded_file = request.FILES['geojson_file']
            
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
                
            # Explicitly close file to avoid issues in the next steps
            temp_file.close()                  
            
            # If a previously generated ZIP exists, remove it before generating a new one
            potential_zip_path = os.path.join(media_path, f"{zip_sufix}_{session_id}.{format}")            
            if os.path.exists(potential_zip_path):    
                os.remove(potential_zip_path)
            
            try:
                output_dir = analysis.run_energy_analysis(temp_file_path)
            except Exception as e:
                log.error(f"[ERROR] Error in analysis: {e}")
                output_dir = None
    
            if output_dir is not None:
                relative_output_dir = os.path.relpath(output_dir, media_path)
                download_path = shutil.make_archive(archive_path, format, media_path, relative_output_dir)
                download_path = os.path.split(download_path)[-1]
                context["download_path"] = f"{settings.MEDIA_URL}{download_path}" 
                
                if os.path.exists(output_dir): 
                    shutil.rmtree(output_dir)   
            else:
                context['error'] = "Processing error, please try again letter!"
                    
            if os.path.exists(temp_file_path):    
                os.remove(temp_file_path)            
            
            # Respond to HTMX by rendering the JSON upload form
            return render(request, "includes/_json_upload_form.html", context=context)
        else:
            context["form"] = form
            # Respond to HTMX by rendering the JSON upload form
            return render(request, "includes/_json_upload_form.html", context=context)
            

    response = render(request, "common/landing.html", context=context)
    return response