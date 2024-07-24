from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('images/', views.display_images, name='display_images'),
    path('records/', views.records, name='records'),
    path('reset/', views.reset, name='reset'),
    path('save/', views.save, name='save')
] 
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)