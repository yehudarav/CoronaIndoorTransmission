import numpy
from unum.units import *
import unum
from  . import abstractAgent

def getRoomClass(modelType):

    father = getattr(abstractAgent,modelType)
    if father is None:
        raise ValueError("modelType can be either AgentEquiDistance or AgentEvents")
    return type('room', (Room,father), {})


class Room(object):
    """
        Implements a room that tracks after the concentration
        of the virus in the air and its concentration on surfaces.

        the surface holds a list of all the spots with their respective concentration.
        When we calculate the amount that a person was exposed we randomize
        (touching frequency/time steps) times the probability to touch each stain.

        Events:

            - cleanFomite
            - social (inter person communication).


    """
    _virusConcentrationAir      = None  # c
    _shedList                  = None  # A list of stains from coughing
    _fomiteConcentration        = None

    _personInRoom = None # a map name->person.

    @property
    def dt(self):
        return self.model.dt

    @property
    def personInRoom(self):
        return self._personInRoom

    @property
    def virusFomiteConcentration(self):
        return self._fomiteConcentration

    @property
    def virusConcentrationAir(self):
        return self._virusConcentrationAir

    @property
    def effectiveSurfaceArea(self):
        return self.settings["physical"]["surfaceArea"]*self.furnitureSurfaceAreaFactor

    @property
    def furnitureSurfaceArea(self):
        return self.settings["physical"]["surfaceArea"]*(self.furnitureSurfaceAreaFactor-1)

    @property
    def furnitureSurfaceAreaFactor(self):
        return self.settings["physical"]["furnitureSurfaceAreaFactor"]

    @property
    def roomVolume(self):
        return self.settings["physical"]["surfaceArea"]*self.settings["physical"]["height"]

    @property
    def personInteractionFrequency(self):
        return self.settings["social"]["interactionfrequency"]

    @property
    def decayRateAir(self):
        return self.settings["air"]["decayRate"]+\
               self.settings["air"]["exchangeRate"]

    @property
    def decayRateSurface(self):
        return self.settings["surface"]["decayRate"]

    @property
    def fomiteSurfaceArea(self):
        return self.settings["fomite"]["surfaceArea"]

    @property
    def decayRateFomite(self):
        return self.settings["fomite"]["decayRate"]

    @property
    def cleaningEfficiencyFomite(self):
        return self.settings["actions"]["cleanFomite"]["efficiency"]

    @property
    def shedList(self):
        return self._shedList

    def __init__(self, unique_id, model):
        super().__init__(unique_id,model,agentType="room")

        self._virusConcentrationAir      = 0/m**3
        self._shedList = []
        self._fomiteConcentration = 0/m**2
        self._personInRoom ={}

        self._fieldChange["airconcentration"] = 0.
        self._fieldChange["fomite"] = 0.
        self._fieldChange["clean_fomite"] = 0.

    def enterRoom(self,person):
        """
            Register person in a room.
        :param person:
        :return:
        """
        self._personInRoom[person.unique_id] = person
        person.enterLocation(self)

    def leaveRoom(self,person):
        self._personInRoom[person.unique_id].leaveLocation()

        del self._personInRoom[person.unique_id]


    def _event_handle_cleanFomite(self):
        self._fieldChange['clean_fomite'] +=  -(self._fomiteConcentration*self.cleaningEfficiencyFomite*self.fomiteSurfaceArea).asNumber()
        self._fomiteConcentration   *=   (1-self.cleaningEfficiencyFomite)

    def _event_handle_social(self):
        """
            For now, assume there are 2 people in the room.
        :return:
        """

        person1 = self.model.primary
        person2 = self.model.secondary

        person1Person2 = (person1.handSurfaceArea * person1.factorHandToFace * (
                          person2.virusHandConcentration - person1.virusHandConcentration)).asNumber()

        person1.updateSocial( person1Person2)
        person2.updateSocial(-person1Person2)


    def updateAir(self,viralLoad):
        self._fieldChange["airconcentration"] += viralLoad

    def updateStain(self,viralLoad,stainArea):
        self._shedList += [dict(stainArea=stainArea,viralLoadSurface=viralLoad,date=self.model.getCurrentDatetime)]

    def  updateFomite(self,viralLoad):
        self._fieldChange['fomite'] += viralLoad

    def step(self):
        """
            For each person:
                1. get shed viruses.
                2. shed to environmental surfaces.
                3. shed to fomite.
                4. shed to air.

            Decay virus in the room.

            Use first order decay, solve implicitly.
            remember that the decayRate is in 1/[dt] units.

        :return:
            None
        """
        decayRateAir,_     = self.decayRateAir.matchUnits(1/self.dt)

        airConcentrationChange = self._fieldChange["airconcentration"]/self.roomVolume
        air_before = self._virusConcentrationAir
        self._virusConcentrationAir = (self._virusConcentrationAir + airConcentrationChange)/ \
                                      (1 + (decayRateAir * self.dt).asNumber())

        fomiteChange = self._fieldChange["fomite"]/self.fomiteSurfaceArea

        fomite_before = self._fomiteConcentration
        decayRateFomite,_    = self.decayRateFomite.matchUnits(1/self.dt)

        self._fomiteConcentration = (self._fomiteConcentration + fomiteChange)/ \
                                    (1 + (decayRateFomite * self.dt).asNumber())

        decayRateSurface, _ = self.decayRateSurface.matchUnits(1 / self.dt)
        for stain in self._shedList:
            stain["viralLoadSurface"] = (stain["viralLoadSurface"])/ \
                                    (1 + (decayRateSurface * self.dt).asNumber())

        self._fieldChange["fomite_with_decay"] = self._fomiteConcentration - fomite_before
        self._fieldChange["air_with_decay"] = self._virusConcentrationAir- air_before

        self.collect()
        self._fieldChange["airconcentration"] = 0.
        self._fieldChange["fomite"] = 0.
        self._fieldChange["clean_fomite"] = 0.

    def collect(self):

        ts = dict(name = self.unique_id,
                  date = self.model.getCurrentDatetime,
                  virusConcentrationAir =  self.virusConcentrationAir,
                  fomiteConcentration=self.virusFomiteConcentration,
                  change_air =self._fieldChange["airconcentration"],
                  change_fomite = self._fieldChange["fomite"],
                  fomite_with_decay = self._fieldChange["fomite_with_decay"],
                  air_with_decay    = self._fieldChange["air_with_decay"],
                  clean_fomite      = self._fieldChange["clean_fomite"])

        self._history.append(ts)
