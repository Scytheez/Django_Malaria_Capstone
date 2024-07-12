from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('images/', views.display_images, name='display_images'),
    path('records/', views.records, name='records'),
]
