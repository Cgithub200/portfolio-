from rest_framework import serializers
from .models import StoredImage


class StoredImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = StoredImage
        fields = 'address','caption'





