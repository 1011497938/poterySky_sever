import json
from django.http import HttpResponse
from .data_process.dataStore import poteryManager, sentenceManager, authorManager
from .data_process.commonFunction import writeJson, loadJson

def jsonHttp(json_object):
    return HttpResponse(json.dumps(json_object))

def init(request):
    data = {'msg': 'init is down', 'success': True}
    return jsonHttp(data)

def getTopPoteriesVec(request):
    start  = request.GET.get('start')
    end  = request.GET.get('end')
    start = int(start)
    end = int(end)

    data = [[potery.id] + potery.getVec3() + [potery.rank]  for potery in poteryManager.poteries[start:end]]
    # print([elm[]])
    # for potery in poteryManager.poteries[start:end]:
    #     data[potery.id] = potery.vec3 + [potery.rank]
    return jsonHttp(data)

param2topPoteriesSims = {}
def getTopPoteriesSims(request):
    start  = request.GET.get('start')
    end  = request.GET.get('end')
    max_edge_per_star = request.GET.get('max_edge')


    key = start + '-' + end + '-' + max_edge_per_star
    if key in param2topPoteriesSims:
        return jsonHttp(param2topPoteriesSims[key])

    start = int(start)
    end = int(end)
    max_edge_per_star = int(max_edge_per_star)

    poteries = poteryManager.poteries[start:end]
    poteries = set(poteries)
    pid2sim = {}
    for potery in poteries:
        sim_poteries = potery.getSimPoteries(1000)
        pid2sim[potery.id] = [elm.id for elm in sim_poteries if elm in poteries][0:max_edge_per_star]
    pid2sim = {pid: pid2sim[pid] for pid in pid2sim if len(pid2sim[pid])!=0}

    param2topPoteriesSims[key] = pid2sim
    return jsonHttp(pid2sim)

def getSimPotery(request):
    p_id  = request.GET.get('p_id')
    potery = poteryManager.get(p_id)
    sim_poteries = potery.getSimPoteries()

    data = {}
    for sim_potery in sim_poteries:
        data[sim_potery.id] = sim_potery.getSimpDict()
    return jsonHttp(data)

def getPoteries(request):
    p_ids = request.GET.get('p_ids')
    p_ids = p_ids.split(',')
    poteries = [poteryManager.get(p_id) for p_id in p_ids]
    data = {potery.id: potery.getSimpDict() for potery in poteries}
    return jsonHttp(data)
