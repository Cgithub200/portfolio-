# Generated by Django 5.0.4 on 2024-04-18 08:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('my_clip', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='storedimage',
            name='address',
            field=models.ImageField(upload_to='my_clip\roco-dataset\\data\train\radiology\\images'),
        ),
    ]
