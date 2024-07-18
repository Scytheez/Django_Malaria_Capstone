from django.db import models
from django.conf import settings
import os

# Create your models here.

class upload_img(models.Model):
    image = models.ImageField(upload_to='uploaded_img/', null=True, blank=True)
    status = models.CharField(max_length=12, null=True, blank=True, default='NULL')
    con_lvl = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True, default=00.00)
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
