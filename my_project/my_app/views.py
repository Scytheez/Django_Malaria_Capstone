from ML_Pred_Malaria_Backend.pred_main import validate, predict

from django.db.models import Count, Q
from django.db import transaction
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from .models import upload_img, SavedRecord, RecordData
from datetime import date
from collections import defaultdict
import json
import shutil
import os

# PDF/CSV Generate Modules
import io
import csv
from django.http import FileResponse, HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

########################### FUNCTIONS
def rm_file(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def custom_404(request, exception):
    return render(request, '404.html')

########################### UPLOAD IMAGE
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

########################### DISPLAY IMAGE
def display_images(request):
    images = upload_img.objects.all()
    paginator = Paginator(images, 15)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    par_count = upload_img.objects.filter(label='parasitized').count()
    un_count = upload_img.objects.filter(label='uninfected').count()

    context = {
        'page_obj': page_obj,
        'par_count': par_count,
        'un_count': un_count,
    }

    return render(request, 'display.html', context)

def remove_img(request, img_id):
    img = upload_img.objects.get(id=img_id)
    # remove from dir
    path = f'media/{img}'
    os.remove(path)

    # remove from model
    img.delete()

    return redirect('display_images')

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

    # update image path in RecordData model
    updated_path = f'records/{dir_name}/'
    update_rec_data = RecordData.objects.filter(image__startswith='media')
    with transaction.atomic():
        for item in update_rec_data:
            record_path = item.image.path
            record_name = os.path.basename(record_path)
            record_full = os.path.join(updated_path, record_name)
            item.image = record_full
            item.save()

    # Removing record from previous model/dir
    upload_img.objects.all().delete()
    
    rm_file('media/parasitized/')
    rm_file('media/uninfected/')
    rm_file('media/uploaded_img/')
    rm_file('media/validation/invalid/')
    rm_file('media/validation/valid/')

    return redirect('records')

########################### RECORDS
def records(request):
    model_record = SavedRecord.objects.all()
    paginator = Paginator(model_record, 6)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    date_value = list(SavedRecord.objects.values_list('date', flat=True).distinct())
    date_value = [date.strftime('%Y-%m-%d') for date in date_value]

    counts = SavedRecord.objects.annotate(
        parasitized_count=Count('record_data', filter=Q(record_data__label='parasitized')),
        uninfected_count=Count('record_data', filter=Q(record_data__label='uninfected'))
    ).values('date', 'parasitized_count', 'uninfected_count')

    parasitized_counts = defaultdict(int)
    uninfected_counts = defaultdict(int)

    for count in counts:
        parasitized_counts[count['date']] += count['parasitized_count']
        uninfected_counts[count['date']] += count['uninfected_count']

    parasitized_counts = list(parasitized_counts.values())
    uninfected_counts = list(uninfected_counts.values())

    context = {
        'page_obj': page_obj,
        'date_value': json.dumps(date_value),
        'parasitized_count': json.dumps(parasitized_counts),
        'uninfected_count': json.dumps(uninfected_counts),
    }

    return render(request, 'records.html', context)

def del_record(request, record_id):
    # delete record in dir
    path = RecordData.objects.filter(record_number_id=record_id).first()
    path = str(path.image)
    part = path.split('/')
    record = part[1]
    try:
        folder_path = 'media/records/'
        if record in os.listdir(folder_path):
            file_path = os.path.join(folder_path, record)
            shutil.rmtree(file_path)
            print(f"Record '{record}' deleted successfully.")
        else:
            print(f"Record '{record}' not found in the folder.")
    except Exception as e:
        print(e)
    # delete record in model
    del_saved = SavedRecord.objects.filter(id=record_id)
    del_data = RecordData.objects.filter(record_number_id=record_id)
    for item in del_saved:
        item.delete()
    for item in del_data:
        item.delete()

    return redirect('records')

########################### VIEW
def view_img(request, record_id):
    record = RecordData.objects.filter(record_number_id=record_id)
    paginator = Paginator(record, 10)
    page_number = request.GET.get("page")
    record_page_obj = paginator.get_page(page_number)
    return render(request, 'view.html', {'record_page_obj': record_page_obj})

# Pdf generator
def pdf_generate(request, record_id):
    get_rec = RecordData.objects.filter(record_number_id=record_id)

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    data = []

    for item in get_rec:
        if item.image and item.image.path and item.label:
            img_url = item.image.path
            lbl = item.label
            data.append((img_url, lbl))

    x_offset = 0.5 * inch
    y_offset = height - 1 * inch
    col_width = (width - 1 * inch) / 5
    row_height = 2 * inch
    gap = 0.2 * inch

    for i, (image_path, label) in enumerate(data):
        if i % 5 == 0 and i != 0:
            y_offset -= row_height
            x_offset = 0.5 * inch

        if i % 25 == 0 and i != 0:
            p.showPage()
            y_offset = height - 1 * inch

        p.drawImage(image_path, x_offset, y_offset - 1 * inch, width=col_width - gap, height=col_width - gap)
        p.drawString(x_offset, y_offset - 1.2 * inch - gap, label)
        x_offset += col_width

    p.showPage()
    p.save()

    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='record.pdf')

def csv_generate(request, record_id):
    get_rec = RecordData.objects.filter(record_number_id=record_id)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'

    writer = csv.writer(response)
    
    writer.writerow(['Prediction', 'Confidence Level'])

    for item in get_rec:
        lbl = item.label
        conf = item.con_lvl
        writer.writerow([lbl, f'{conf}%'])

    return response