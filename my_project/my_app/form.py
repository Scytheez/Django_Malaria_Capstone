from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'allow_multiple_file': True}))
