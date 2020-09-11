from pathlib import Path
from agentsimulation.model import getModelClass
from agentsimulation.person import EXPOSED
from hera import datalayer
import os
import sys
import json
import pandas

def run(i,jsonObj,name):

    projectName = "Corona_singleRoom_withTalk"
    maxRuns = jsonObj['simulation']['maxRuns']

    del jsonObj['simulation']['maxRuns']
    casequery = datalayer.dictToMongoQuery(jsonObj, prefix="params")

    docList = datalayer.Simulations.getDocuments(projectName=projectName,data="agents" ,**casequery)
    print(len(docList))

    if len(docList) >= maxRuns:
       return

    modelCls = getModelClass(jsonObj)

    model = modelCls(jsonObj,i+len(docList))
    model.runSimulation(jsonObj['simulation']['terminatePrimaryInfected'])

    primary = model.primary.history(unitless=True).assign(agent="primary")
    secondary = model.secondary.history(unitless=True).assign(agent="secondary")
    room = model.room.history(unitless=True)

    individual_units = model.primary.hisoryUnits
    room_units = model.room.hisoryUnits

    try:
        serialIndex = secondary.iloc[-1].symptomsAppear-primary.iloc[-1].symptomsAppear
        serialIndexLength = serialIndex.total_seconds()
    except TypeError:
        serialIndexLength = None

    try:
        infectionDateDiff = secondary.iloc[-1].incubationStart -primary.iloc[-1].symptomsAppear
        infectionDateDiff_sec = infectionDateDiff.total_seconds()
    except TypeError:
        infectionDateDiff_sec = None

    agents = pandas.concat([primary,secondary],ignore_index=True,sort=False)



    documentType = "coronaAgent"
    descAgents = dict(
         runid=i,
         data="agents",
         primaryState=primary.iloc[-1].state,
         secondaryState=secondary.iloc[-1].state,
         secondarySick=secondary.iloc[-1].state == EXPOSED,
         serialIndex=serialIndexLength,
         infectionDateDiff=infectionDateDiff_sec,
         params = jsonObj,
         individual_units=individual_units,
         room_units=room_units
     )

    obj = datalayer.Simulations.addDocument(projectName=projectName,
                                      resource="",
                                      dataFormat=datalayer.datatypes.PARQUET,
                                      type=documentType,
                                      desc = descAgents)

    basePath = os.path.join("results_data3",name,"run_%s" % str(obj.id))
    Path(basePath).mkdir(parents=True, exist_ok=True)
    agent_path = os.path.join(basePath,"agents.parquet")
    obj.resource = os.path.abspath(agent_path)
    obj.save()

    descRoom = descAgents
    descRoom['data'] = "room"

    room_path = os.path.abspath(os.path.join(basePath,"room.parquet"))
    obj = datalayer.Simulations.addDocument(projectName=projectName,
                                      resource=room_path,
                                      dataFormat='parquet',
                                      type=documentType,
                                      desc = descRoom)

    if jsonObj['simulation']['collectFullData']:
        agents.to_parquet(agent_path, use_deprecated_int96_timestamps=True, compression='gzip')
        room.to_parquet(room_path, use_deprecated_int96_timestamps=True, compression='gzip')

    return model


def updateConf(base,newconf):

    def _updateConf(base,newconf):
        for k,v in newconf.items():
            if isinstance(v,dict):
                _updateConf(base[k],newconf[k])
            else:
                base[k] = newconf[k]
    _updateConf(base,newconf)

if __name__=="__main__":

    with open("configuration/runningConf.json") as file:
        base = json.load(file)

    with open(os.path.join("configuration",sys.argv[1])) as file:
        conf = json.load(file)

    updateConf(base,conf)
    name = sys.argv[1].split(".")[0]
    model = run(int(sys.argv[2]),base,name)
