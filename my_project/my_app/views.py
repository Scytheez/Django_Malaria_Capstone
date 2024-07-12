from django.shortcuts import render, redirect
from .form import ImageUploadForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage

#from ML_Pred_Malaria_Backend.pred_main import 

#   Base Upload Webpage
def upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('image')
            fs = FileSystemStorage()
            image_urls = []
            for image in images:
                filename = fs.save(image.name, image)
                image_urls.append(fs.url(filename))
            return render(request, 'display.html', {'images': image_urls})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html')

#   Display Image
def display_images(request):
    fs = FileSystemStorage()
    files = fs.listdir(settings.MEDIA_ROOT)[1]  # get list of files in media directory
    image_urls = [fs.url(file) for file in files]
    return render(request, 'display.html', {'images': image_urls})

#   Records
def records(request):
    return render(request, 'records.html')