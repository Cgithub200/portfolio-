from django.db import models
from django.forms import CharField
from django.contrib.postgres.fields import ArrayField
# Create your models here.

def default_array():
    return [0] * 556
class StoredImage(models.Model):
    address = models.CharField(max_length=200, default='')
    caption = models.CharField(max_length=300, default='')
    image_data = models.JSONField(default=default_array)
    def __str__(self):
        return (self.address)
    

class StoredInternalImage(models.Model):
    address = models.CharField(max_length=200, default='')
    caption = models.CharField(max_length=300, default='')
    image_data = models.JSONField(default=default_array)
    def __str__(self):
        return (self.address)