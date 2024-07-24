from ML_Pred_Malaria_Backend.pred_main import validate, predict

from django.db import transaction
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from .models import upload_img, SavedRecord, RecordData
from datetime import date
import shutil
import os

##### FUNCTIONS
def rm_file(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

##### UPLOAD IMAGE
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

##### DISPLAY IMAGE
def display_images(request):
    images = upload_img.objects.all()
    paginator = Paginator(images, 18)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, 'display.html', {'page_obj': page_obj})

def reset(request):
    upload_img.objects.all().delete()
    
    rm_file('media/parasitized/')
    rm_file('media/uninfected/')
    rm_file('media/uploaded_img/')
    rm_file('media/validation/invalid/')
    rm_file('media/validation/valid/')

    return redirect('upload')

def save(request):
    path = "media/records/"
    name = "record"

    i = 1
    while True:
        dir_name = f"{name} {i}"
        full_path = os.path.join(path, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Directory '{full_path}' created.")
            break
        i += 1

    new_path = f'media/records/{dir_name}/'
    update_records = upload_img.objects.filter(status='valid')

    # Update image paths in update_records
    with transaction.atomic():
        for item in update_records:
            record_path = item.image.path
            record_name = os.path.basename(record_path)
            record_full = os.path.join(new_path, record_name)
            item.image = record_full
            item.save()

    # Move images to the new directory
    old_paths = ['media/uninfected/', 'media/parasitized/']

    for old_path in old_paths:
        for filename in os.listdir(old_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                source_file = os.path.join(old_path, filename)
                destination_file = os.path.join(new_path, filename)
                shutil.move(source_file, destination_file)

    # Insert data into RecordData
    record_number = i
    dt = date.today()
    
    with transaction.atomic():
        saved_record = SavedRecord(record_number=record_number, date=dt)
        saved_record.save()

        for item in update_records:
            image = item.image
            label = item.label
            con_lvl = item.con_lvl
            insert = RecordData(image=image, label=label, con_lvl=con_lvl, record_number=saved_record)
            insert.save()

    # Removing record from previous model/dir
    upload_img.objects.all().delete()
    
    rm_file('media/parasitized/')
    rm_file('media/uninfected/')
    rm_file('media/uploaded_img/')
    rm_file('media/validation/invalid/')
    rm_file('media/validation/valid/')

    return redirect('records')

##### RECORDS
def records(request):
    return render(request, 'records.html')
