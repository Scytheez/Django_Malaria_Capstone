from django.db import models
from django.conf import settings
import os

# Create your models here.

class upload_img(models.Model):
    image = models.ImageField(upload_to='uploaded_img/', null=True, blank=True)
    status = models.CharField(max_length=12, null=True, blank=True, default='NULL')
    con_lvl = models.CharField(max_length=5, null=True, blank=True, default='NULL')
    label = models.CharField(max_length=12, null=True, blank=True, default='NULL')

    def move_image(self, new_directory):
        old_path = self.image.path
        new_path = os.path.join(settings.MEDIA_ROOT, new_directory, os.path.basename(old_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)
        self.image.name = os.path.join(new_directory, os.path.basename(old_path))
        self.save()
        if os.path.exists(old_path):
            os.remove(old_path)

    def __str__(self):
        return str(self.image)

class SavedRecord(models.Model):
    record_number = models.IntegerField(null=True, blank=True)
    date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"Record {self.record_number} - {self.date}"

class RecordData(models.Model):
    record_number = models.ForeignKey(SavedRecord, on_delete=models.CASCADE, related_name='record_data')
    image = models.ImageField(null=True, blank=True)
    con_lvl = models.CharField(max_length=5, null=True, blank=True)
    label = models.CharField(max_length=12, null=True, blank=True)

    def __str__(self):
        return f"Data for Record {self.record_number.record_number} - {self.label}"