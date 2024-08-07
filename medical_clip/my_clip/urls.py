import django
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('add_image/', views.Expand_dataset, name='add_image'),
    path('add_image', views.Expand_dataset, name='add_image'),
    path('ret_image/', views.ret_image, name='ret_image'),
    path('ret_image', views.ret_image, name='ret_image'),
    #path('update_database/', views.update_database, name='update_database'),
    #path('update_database', views.update_database, name='update_database'),
    path("",views.login,name="login"),
] + static(settings.STATIC_URL,document_root=settings.STATICFILES_DIRS)

