from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('images/', views.display_images, name='display_images'),
    path('remove_img/<int:img_id>/', views.remove_img, name='remove_img'),
    path('records/', views.records, name='records'),
    path('reset/', views.reset, name='reset'),
    path('save/', views.save, name='save'),
    path('confm/', views.confm, name='confm'),
    path('view/<int:record_id>', views.view_img, name='view_img'),
    path('delete/<int:record_id>', views.del_record, name='del_record'),
    path('pdf_generate/<int:record_id>/', views.pdf_generate, name='pdf_generate'),
    path('csv_generate/<int:record_id>/', views.csv_generate, name='csv_generate'),
] 
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
