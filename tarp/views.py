from django.shortcuts import render
from .algorithms import ytres
from .forms import LinkForm
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@csrf_exempt
def index(request):
    form = LinkForm()
    if request.method == 'POST':
        form = LinkForm(request.POST)
        if form.is_valid():
            link = form.cleaned_data['link']
            doubts , requests , Statements , negative = ytres(link)

        return render(request,'test.html',{"form":form,"doubts":doubts,"requests":requests,"Statements":Statements,"negative":negative}) 
    else:
        return render(request,'test.html',{"form":form})