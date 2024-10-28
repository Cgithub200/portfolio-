from django.contrib import admin

# Register your models here.

from .models import StoredImage,StoredInternalImage
admin.site.register(StoredImage)
admin.site.register(StoredInternalImage)