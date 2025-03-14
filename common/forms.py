from django import forms
import os

def validate_file_size(file, max_size_mb=5):
    """Validate file size (default: 5MB limit)"""
    max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
    if file.size > max_size:
        raise forms.ValidationError(f"File size should not exceed {max_size_mb}MB.")

class GeoJsonUploadForm(forms.Form):
    
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop("request", None)
        super().__init__(*args, **kwargs)

    geojson_file = forms.FileField(
        label="Upload GeoJSON File",
        help_text="Supported format: .json (Max size: 5MB)",     
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control form-control-lg w-100',
            'accept': ".json"
        })      
    )
    
    def clean(self):
        cleaned_data = super().clean()
    
        if self.request and not self.request.user.is_authenticated:         
            raise forms.ValidationError("You must logged in to upload file!")
        return cleaned_data

    def clean_geojson_file(self):
        file = self.cleaned_data.get("geojson_file")
        
        if file:
            # Validate file extension
            ext = os.path.splitext(file.name)[1].lower()
            if ext != ".json":
                raise forms.ValidationError("Only JSON files are allowed.")

            # Validate MIME type
            if not file.content_type.startswith("application/"):
                raise forms.ValidationError("Invalid MIME type. Please upload a valid JSON file.")

            # Validate file size
            validate_file_size(file)

        return file
