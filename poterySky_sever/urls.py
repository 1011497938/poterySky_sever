from django.contrib import admin
from django.urls import path
# from . import func
# from . import pre_process
from . import funcV2
urlpatterns = [
    # path('init/', func.init),
    # path('topPoteriesVec/', func.getTopPoteriesVec),
    # path('topPoteriesSim/', func.getTopPoteriesSims),
    # path('simPoteries/', func.getSimPotery),
    # path('poteries/', func.getPoteries),
    # path('processData/', pre_process.processData)
    
    path('getSomePoteries/', funcV2.getSomePoteries),
    path('getSomeAuthors/', funcV2.getSomeAuthors),
    path('getPoteryInfo/', funcV2.getPoteryInfo),
    path('getAuthorInfo/', funcV2.getAuthorInfo),
    path('getSomeWords/', funcV2.getSomeWords),
    path('getRelatedWords/', funcV2.getRelatedWords),
    path('analyzeWritePotery/', funcV2.analyzeWritePotery),
    path('analyzePotery/', funcV2.analyzePotery),
]
