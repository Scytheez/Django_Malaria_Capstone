from ML_Pred_Malaria_Backend.pred_main import validate, predict

from django.shortcuts import render, redirect
from .models import upload_img

# Upload Webpage
def upload(request):
    if request.method == 'POST':
        images = request.FILES.getlist('image')
        for image in images:
            upload_img.objects.create(image=image)
        
        # validate images
        validate()

        # predict images
        predict()

        return redirect('display_images')
    else:
        return render(request, 'upload.html')

# Display Image
def display_images(request):
    images = upload_img.objects.all()
    return render(request, 'display.html', {'images': images})

# Records
def records(request):
    return render(request, 'records.html')