#<form method="post" class="my-form-class">

from django import forms

class LinkForm(forms.Form):
  link = forms.CharField(label='Link', max_length=200, widget=forms.TextInput(attrs={'class': 'my-input-class'}))