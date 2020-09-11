import numpy
from unum.units import *
from  . import abstractAgent
from scipy.stats import uniform,gamma,beta

from . import log_interp1d,ml,SUSCEPTIBLE,EXPOSED,INFECTED,RECOVERED

import pandas


def getPersonClass(modelType):

    father = getattr(abstractAgent,modelType)
    if father is None:
        raise ValueError("modelType can be either AgentEquiDistance or AgentEvents")
    return type('person', (Person,father), {})


class Person(object):
    """
        A person agent.

        can be either susceptible, exposed, infected or recovered.

        - Susceptible:
            - keep track of the hand dynamics
            - keep track to the amount that was currently exposed
              from : fomite, environment (breath and touch) and interperson interaction.

        - Exposed:
            - determine incubation.
            - increase the viral load until incubation reached.

        - Infected: [not implemented]
            -  random recovery time.
            -  decrease virues during this time.


    Events:

        - touch fomite
        - touch face
        - touch surface
        - touch person.
        - cough
        - sneeze
        - wash  hands
        - immune system.


    """

    _currentState       = None
    _currentLocation    = None      # a ref to the compartment the person resides in.
    _residenceTime      = None      # the time to stay in this compartment.
    _residenceTimeCounter = None    # The time the person stayed in that compartment.

    _viralLoad          = None      # The viral load when exposed.

    _currentExposure    = None      # The current exposure to the virus
    _totalExposure      = None      # The total exposure to the virus.
    _virusHandConcentration = None  # Virus concentration on hands.

    _incubationStartDatetime = None
    _incubationPeriod = None

    _maxViralLoad  = None
    _nonEvaporatingDropletsVolume_cough = None
    _evaporatingDropletsVolume_cough = None

    _nonEvaporatingDropletsVolume_sneeze = None
    _evaporatingDropletsVolume_sneeze = None


    def enterLocation(self,locationRef):
        """
            The person enters the location
        :param locationRef: room obj.
            The object that the person enters to.
        :return:
        """
        self._currentLocation = locationRef

        # if locationRef.unique_id in self.settings["residence"]:
        #     residencefunc = self.settings["residence"][locationRef.unique_id]
        #     self._residenceTime = getattr(self, "func_%s" % residencefunc["name"])(**residencefunc["params"])
        #     self._residenceTimeCounter = 0*min
        # else:
        #     self._residenceTime = None
        #     self._residenceTimeCounter = None

    def leaveLocation(self):
        self._currentLocation = None


    @property
    def location(self):
        return self._currentLocation

    @property
    def currentState(self):
        return self._currentState

    @property
    def washingHandEfficiency(self):
        return self.settings["actions"]["washHands"]["efficiency"]

    @property
    def nonEvaporatingDropletsVolume_cough(self):
        return self._nonEvaporatingDropletsVolume_cough

    @property
    def evaporatingDropletsVolume_cough(self):
        return self._evaporatingDropletsVolume_cough

    @property
    def nonEvaporatingDropletsVolume_sneeze(self):
        return self._nonEvaporatingDropletsVolume_sneeze

    @property
    def evaporatingDropletsVolume_sneeze(self):
        return self._evaporatingDropletsVolume_sneeze

    @property
    def nonEvaporatingDropletsVolume_talk(self):
        return self._nonEvaporatingDropletsVolume_talk

    @property
    def evaporatingDropletsVolume_talk(self):
        return self._evaporatingDropletsVolume_talk

    @property
    def currentDatetime(self):
        return self.model.getCurrentDatetime

    @property
    def incubationStartDatetime(self):
        return self._incubationStartDatetime


    @property
    def incubationPeriod(self):
        return self._incubationPeriod

    @property
    def virusHandConcentration(self):
        return self._virusHandConcentration

    @property
    def incubationEnd(self):
        return self._incubationStartDatetime + self._incubationPeriod


    @property
    def viralLoad(self):
        return self._viralLoad

    @property
    def totalExposed(self):
        return self._totalExposure

    @property
    def currentExposure(self):
        return self._currentExposure


    @property
    def dt(self):
        return self.model.dt

    @property
    def viralLoadFactor_cough(self):
        return self.model.settings["person"]["actions"]["cough"]["viralLoadFactor"]

    @property
    def viralLoadFactor_sneeze(self):
        return self.model.settings["person"]["actions"]["sneeze"]["viralLoadFactor"]

    @property
    def viralLoadFactor_talk(self):
        return self.model.settings["person"]["actions"]["talk"]["viralLoadFactor"]

    @property
    def maxViralLoad(self):
        return self.settings["physiology"]["maxviralload"]


    @property
    def sicknessPeriod(self):
        return self.settings["physiology"]["sickness"]["period"]

    @property
    def sicknessPeriod_datetime(self):
        return pandas.to_timedelta("%sd" % self.sicknessPeriod.asNumber(d))

    @property
    def factorSurfaceToHand(self):
        """
            f12
        :return:
        """
        return self.settings["actions"]["touchSurface"]["factorSurfaceToHand"]

    @property
    def factorHandToSurface(self):
        """
            f12
        :return:
        """
        return self.settings["actions"]["touchSurface"]["factorHandToSurface"]


    @property
    def handSurfaceArea(self):
        return self.settings["physiology"]["hand"]["surfaceArea"]

    @property
    def factorHandToFace(self):
        """
            any tissue to tissue.
            f23
        :return:
        """
        return self.settings["actions"]["touchFace"]["factorHandToFace"]

    @property
    def autoincolationVolume(self):
        return self.settings["actions"]["touchFace"]["autoincolationVolume"]

    @property
    def handToMouth(self):
        return self.settings["actions"]["touchFace"]["handToMouth"]

    @property
    def breathingRate(self):
        return self.settings["physiology"]["breathingRate"]

    @property
    def breathingEfficiency(self):
        return self.settings["physiology"]["breathingEfficiency"]


    @property
    def handDecayRate(self):
        return self.settings["physiology"]["hand"]["decayRate"]

    def __init__(self, unique_id, model,startState=SUSCEPTIBLE):
        """
            Initializes an object.

            Takes the sheding function and parameters from the model.settings[self._agentType]["sheddingFunction"]

                model.settings[self._agentType]["sheddingFunction"] structure is
                {
                    "name" : function name
                    "parameters" : {dictionary with function parameters}
                }


        :param unique_id:
        :param model:
                a scenario object.
        """
        super().__init__(unique_id, model,agentType="person")
        self._currentState = startState
        self._currentLocation = None
        incubationStartDatetime = model.simulationStart if startState == EXPOSED else None

        self._incubationStartDatetime=incubationStartDatetime
        self._incubationPeriod= self.get_incubationPeriod()

        self._dt = model.settings["simulation"]["dt"]
        self._viralLoad = 0/ml
        self._virusHandConcentration = 0/cm**2
        self._currentExposure = 0.
        self._totalExposure = 0.
        self._fieldChange["surfaceToHand"] = 0.
        self._fieldChange["fomiteToHand"]  = 0.
        self._fieldChange["hand_interperson"] = 0.
        self._fieldChange["faceToHand"] = 0.
        self._fieldChange["wash_hands"] =0.
        self._fieldChange["exposeFromBreath"] = 0.
        self._fieldChange["exposeFromHand"]  = 0.

        self._fieldChange["totalExposeFromBreath"] = 0
        self._fieldChange["totalExposeFromHand"] = 0

        self._fieldChange["expulsion_breath_talk"] = 0.
        self._fieldChange["expulsion_breath_sneeze"] = 0.
        self._fieldChange["expulsion_breath_cough"] = 0.


        self.setExhaleVolume()


    def step(self):

        room = self.location

        self._fieldChange["exposeFromBreath"] = (room.virusConcentrationAir * \
                                             self.breathingRate * \
                                             self.breathingEfficiency * self.dt).asNumber()

        self._currentExposure +=  self._fieldChange["exposeFromHand"] + self._fieldChange["exposeFromBreath"]
        if (self._currentExposure < 0):
            self._currentExposure = 0

        self._totalExposure += self._fieldChange["exposeFromHand"] + self._fieldChange["exposeFromBreath"]
        if (self._totalExposure < 0):
            self._totalExposure = 0

        self._fieldChange["totalExposeFromBreath"] += self._fieldChange["exposeFromBreath"]
        self._fieldChange["totalExposeFromHand"]   += self._fieldChange["exposeFromHand"]

        handchange = (self._fieldChange["surfaceToHand"]     + self._fieldChange["fomiteToHand"]  + \
                      self._fieldChange["hand_interperson"] + self._fieldChange["faceToHand"])/self.handSurfaceArea

        hand_before  = self._virusHandConcentration
        self._virusHandConcentration = (self._virusHandConcentration + handchange)/(1+ (self.handDecayRate*self.dt).asNumber())
        if (self._virusHandConcentration < 0/cm**2):
            self._virusHandConcentration = 0/cm**2

        self._fieldChange["hand_with_decay"] = self._virusHandConcentration - hand_before

        self.collect()

        self._fieldChange["surfaceToHand"] = 0.
        self._fieldChange["fomiteToHand"]  = 0.
        self._fieldChange["hand_interperson"] = 0.
        self._fieldChange["faceToHand"] = 0.
        self._fieldChange["exposeFromBreath"] = 0.
        self._fieldChange["exposeFromHand"]  = 0.
        self._fieldChange["wash_hands"] = 0.
        self._fieldChange["expulsion_breath_talk"] = 0.
        self._fieldChange["expulsion_breath_sneeze"] = 0.
        self._fieldChange["expulsion_breath_cough"] = 0.

        if self.currentState == EXPOSED:
            if self.model.getCurrentDatetime > self.incubationEnd:
                self._currentState = INFECTED
                self._viralLoad = self.maxViralLoad
            else:

                minViral = self.model.settings['person']['physiology']['minviralload'].asNumber()

                # update the viral load
                incubationStartDatetime = self.incubationStartDatetime
                currentdate             = self.model.getCurrentDatetime
                hoursIncubating = (currentdate-incubationStartDatetime).total_seconds()/pandas.to_timedelta("1h").total_seconds()
                totalIncubationPeriod = self.incubationPeriod
                totalIncubationHours = totalIncubationPeriod.total_seconds() / pandas.to_timedelta("1h").total_seconds()

                viralInterpolator = log_interp1d([0,numpy.ceil(totalIncubationHours)],[minViral,self.maxViralLoad.asNumber(1/ml)])
                self._viralLoad = viralInterpolator(hoursIncubating)/ml

        elif self.currentState == INFECTED:
            if (self.model.getCurrentDatetime - self.incubationEnd) > self.sicknessPeriod_datetime:
                self._currentState = RECOVERED
            else:
                ## reduce the viral load with time.
                incubationEnd = self.incubationEnd
                currentdate = self.model.getCurrentDatetime
                hoursSick = (currentdate - incubationEnd).total_seconds() / pandas.to_timedelta("1h").total_seconds()
                if hoursSick < 0:
                    hoursSick =0

                totalHoursSick = self.sicknessPeriod_datetime.total_seconds() / pandas.to_timedelta("1h").total_seconds()

                viralInterpolator = log_interp1d([0, numpy.ceil(totalHoursSick)],
                                                 [self.maxViralLoad.asNumber(1 / ml),1e-10])

                self._viralLoad = viralInterpolator(hoursSick) / ml

        if self._residenceTime is not None:
            if self._residenceTimeCounter > self._residenceTime:
                gotRoom = self.settings

            else:
                self._residenceTimeCounter += self._dt

    def _event_handle_touchFomite(self):
        """
            transfer to hand from fomite of room

            sets
                - hand_fomite

                room.change_fomite
        :return:
            None
        """
        room = self.location

        fomiteToHand = self.factorSurfaceToHand * self.handSurfaceArea * room.virusFomiteConcentration
        handToFomite = self.factorHandToSurface * self.handSurfaceArea * self.virusHandConcentration

        self._fieldChange["fomiteToHand"] += (fomiteToHand - handToFomite).asNumber()
        room.updateFomite(-self._fieldChange["fomiteToHand"])

    def _event_handle_touchFace(self):
        """
            transfer to hand from fomite of room

            sets
                - hand_face
                - expose_hand
        :return:
            None
        """
        handToFace = self.factorHandToFace * self.handSurfaceArea * self.virusHandConcentration
        faceToHand = self.factorHandToFace * self.handSurfaceArea * (self.autoincolationVolume * self.viralLoad / self.handSurfaceArea)

        self._fieldChange["faceToHand"]   += (faceToHand - handToFace).asNumber()
        self._fieldChange["exposeFromHand"] -= (self._fieldChange["faceToHand"]*self.handToMouth)

    def _event_handle_touchSurface(self):
        """
            transfer to hand from fomite of room

            sets
                - hand_surface
        :return:
            None
        """
        surfaceToHand = 0

        for stain in self.location.shedList:
            # stain area on furniture = stainArea * (area of furniture/total area)  (total is effective here).
            # prob to touch stain on furniture = [stain area on furniture]/[area of furniture] = stainArea / total area
            P     = (stain["stainArea"]/self.location.effectiveSurfaceArea).asNumber()
            if self.random.uniform(0,1) < P:
                surfaceToHand = (stain["viralLoadSurface"]*
                                  self.factorSurfaceToHand*
                                  self.handSurfaceArea/stain["stainArea"]).asNumber()
                break # touch 1 stain at a time.

        self._fieldChange["surfaceToHand"] += surfaceToHand

    def _event_handle_cough(self):
        """
            cough and contaminate room air and surfaces.

            set
                room.air
                room.stainList.
        :return:
            None
        """
        stainArea = self.settings["actions"]["cough"]["stainArea"]

        viralExpolsionAir = (self.viralLoadFactor_cough*self.viralLoad * self.evaporatingDropletsVolume_cough).asNumber()
        viralExpolsionSurface = (self.viralLoadFactor_cough*self.viralLoad * self.nonEvaporatingDropletsVolume_cough).asNumber()

        if viralExpolsionAir > 0:
            self.location.updateAir(viralExpolsionAir)

        if viralExpolsionSurface > 0:
            self.location.updateStain(viralExpolsionSurface,stainArea)

        self._fieldChange["expulsion_breath_cough"] += viralExpolsionAir

    def _event_handle_talk(self):
        """
            cough and contaminate room air and surfaces.

            set
                room.air
                room.stainList.
        :return:
            None
        """
        stainArea = self.settings["actions"]["cough"]["stainArea"]

        viralExpolsionAir = (self.viralLoadFactor_talk*self.viralLoad * self.evaporatingDropletsVolume_talk).asNumber()
        viralExpolsionSurface = (self.viralLoadFactor_talk*self.viralLoad * self.nonEvaporatingDropletsVolume_talk).asNumber()


        if viralExpolsionAir > 0:
            self.location.updateAir(viralExpolsionAir)

        if viralExpolsionSurface > 0:
            self.location.updateStain(viralExpolsionSurface,stainArea)

        self._fieldChange["expulsion_breath_talk"] += viralExpolsionAir

    def _event_handle_sneeze(self):
        """
            cough and contaminate room air and surfaces.

            set
                room.air
                room.stainList.
        :return:
            None
        """
        stainArea = self.settings["actions"]["cough"]["stainArea"]

        viralExpolsionAir = (self.viralLoadFactor_sneeze*self.viralLoad * self._evaporatingDropletsVolume_sneeze).asNumber()
        viralExpolsionSurface = (self.viralLoadFactor_sneeze*self.viralLoad * self._nonEvaporatingDropletsVolume_sneeze).asNumber()

        if viralExpolsionAir> 0:
            self.location.updateAir(viralExpolsionAir)

        if viralExpolsionSurface >0:
            self.location.updateStain(viralExpolsionSurface,stainArea)

        self._fieldChange["expulsion_breath_sneeze"] += viralExpolsionAir

    def _event_handle_washHands(self):
        """
            Clear the hand by a certain fraction.

            set
                hand_wash


        :return:
        """
        self._fieldChange["wash_hands"] += -(self._virusHandConcentration*self.washingHandEfficiency*self.handSurfaceArea).asNumber()
        self._virusHandConcentration *= (1 - self.washingHandEfficiency)

    def _event_handle_immuneSystem(self):
        """
            Check if exposed.
            if it is change th state.

            set
                viralLoad.

        :return:
            None
        """
        if self.currentState == SUSCEPTIBLE:
            doseresponse = self.settings["actions"]["immuneSystem"]["doseresponse"]
            doseresponseFunc = getattr(self,"doseresponse_%s" % doseresponse["name"])

            P = doseresponseFunc(exposure=self.currentExposure, **doseresponse["params"])
            val = self.random.uniform(0,1)
            #print("testing for sickness %s: %s %s" % (self.currentExposure,P,val))
            if  val < P:
                # Became infected.
                self._incubationStartDatetime = self.model.getCurrentDatetime
                self._currentState = EXPOSED

            else:
                self._fieldChange["immuneSystem"] = -self.currentExposure
                self._currentExposure = 0

    def updateSocial(self,viralLoad):
        self._fieldChange["hand_interperson"] += viralLoad

    def collect(self):


        ts = dict(name = self.unique_id,
                  state = self.currentState,
                  date = self.model.getCurrentDatetime,
                  viralLoad = self.viralLoad,
                  totalExposure =  self.totalExposed,
                  currentExposure=self.currentExposure,
                  handconcentration=self.virusHandConcentration,
                  incubationStart = self.incubationStartDatetime
                  )

        if self.incubationStartDatetime is not None:
            ts['symptomsAppear'] = self.incubationStartDatetime+self.incubationPeriod
        else:
            ts['symptomsAppear'] = pandas.NaT

        for k,v in self._fieldChange.items():
            ts[k] =v

        self._history.append(ts)

    def doseresponse_exp(self, exposure, k):
        """
            Return the probability to become sick following exposure to a viralLoad

            the function is 1-exp(-viralLoad/k)

        :param self:
        :param viralLoad:
        :param kwargs:
                - k : the dose response coefficient
        :return:
        """

        return 1-numpy.exp(-exposure/k)

    ##
    ## ===================== incubation
    ##
    def get_incubationPeriod(self):
        incubationfunc = self.settings["physiology"]["incubation"]
        return getattr(self, "func_%s" % incubationfunc["name"])(**incubationfunc["params"])


    ################################################################################
    ##########################################################################
    ##      Coughing/sneezing.
    ##

    def setExhaleVolume(self):
        coughDistributionName = self.settings["actions"]["cough"]["dropletModel"]
        evaporatingCough,nonEvaporatingCough=  getattr(self, "_Cough_%s" % coughDistributionName)()

        coughDistributionName = self.settings["actions"]["sneeze"]["dropletModel"]
        evaporatingSneeze,nonEvaporatingSneeze =  getattr(self, "_Sneeze_%s" % coughDistributionName)()

        talkDistributionName = self.settings["actions"]["talk"]["dropletModel"]
        evaporatingTalk,nonEvaporatingTalk=  getattr(self, "_Talk_%s" % talkDistributionName)()

        self._evaporatingDropletsVolume_cough     = evaporatingCough
        self._evaporatingDropletsVolume_sneeze    = evaporatingSneeze
        self._evaporatingDropletsVolume_talk      = evaporatingTalk

        self._nonEvaporatingDropletsVolume_cough  = nonEvaporatingCough
        self._nonEvaporatingDropletsVolume_sneeze = nonEvaporatingSneeze
        self._nonEvaporatingDropletsVolume_talk = nonEvaporatingTalk


    ##
    ## ===================== Cough/sneeze droplet size and volume.
    ##
    def _Cough_Chen(self):
        """
            Distribution fitted from Chen.

        :return:
        """

        # Small droplets.  < 10micron
        nparticles_small = 230
        k = 3.75
        D_small = numpy.arange(0, 20, 0.1)
        Fcdf_small = gamma.cdf(D_small, 3.75)

        Dsmall_avg = (D_small[:-1] + D_small[1:]) / 2.
        F_small = numpy.diff(Fcdf_small)

        Volume_small = (4.0 / 3.0) * numpy.pi * (Dsmall_avg * 1e-6) ** 3 * F_small * nparticles_small
        vol_small_ml = (Volume_small.sum() * m ** 3).asUnit(ml)

        # medium droplets 10micron < x < 225 micron
        # upto 100 evaporates in air.
        nparticles_medium = 210
        params = dict(a=0.2, b=1, loc=53, scale=200)
        D_medium = numpy.arange(10, 100, 1)
        Fcdf_medium = beta.cdf(D_medium, **params)
        Dmedium_avg = (D_medium[:-1] + D_medium[1:]) / 2.
        F_medium = numpy.diff(Fcdf_medium)

        Volume_small = (4.0 / 3.0) * numpy.pi * (Dmedium_avg * 1e-6) ** 3 * F_medium * nparticles_medium
        vol_medium_ml = (Volume_small.sum() * m ** 3).asUnit(ml)

        evaporatingDropletsVolume = vol_small_ml + vol_medium_ml

        ##### == None evaporating
        D_medium = numpy.arange(100, 225, 1)
        Fcdf_medium = beta.cdf(D_medium, **params)
        Dmedium_avg = (D_medium[:-1] + D_medium[1:]) / 2.
        F_medium = numpy.diff(Fcdf_medium)

        Volume_small = (4.0 / 3.0) * numpy.pi * (Dmedium_avg * 1e-6) ** 3 * F_medium * nparticles_medium
        vol_medium_ml = (Volume_small.sum() * m ** 3).asUnit(ml)

        nparticles_large = 20
        D_large = numpy.arange(225,800,1)
        Fcdf_large = uniform.cdf(D_large,loc=225,scale=800-225)
        Dlarge_avg = (D_large[:-1]+D_large[1:])/2.
        F_large    = numpy.diff(Fcdf_large)

        Volume_small = (4.0/3.0)*numpy.pi*(Dlarge_avg*1e-6)**3*F_large*nparticles_large
        vol_large_ml = (Volume_small.sum()*m**3).asUnit(ml)

        nonEvaporatingDropletsVolume =  vol_large_ml  + vol_medium_ml

        return evaporatingDropletsVolume,nonEvaporatingDropletsVolume


    def _Cough_Nicas(self):
        """
            Following nicas 2007

        :return:
        """
        totalVolume = 0.044*ml
        evaporatingDropletsVolume = 0.01*totalVolume
        nonEvaporatingDropletsVolume = 0.99*totalVolume

        return evaporatingDropletsVolume, nonEvaporatingDropletsVolume

    def _Cough_NicasChen(self):
        """
            The volume is average of Chen and Nicas.

        :return:
        """
        totalVolume = (0.044*ml+0.015*ml)/2.
        evaporatingDropletsVolume = 0.01*totalVolume
        nonEvaporatingDropletsVolume = 0.99*totalVolume

        return evaporatingDropletsVolume,nonEvaporatingDropletsVolume

    def _Sneeze_Chen(self):
        """
                The distribution was taken from the Chen et al. paper.
        :return:
        """
        d = numpy.arange(1, 60, 1)
        Pdist = 2123 + 367734 * numpy.exp(-0.5 * ((numpy.log(d / 7.11) / 0.65)) ** 2)

        Volume = (4.0 / 3.0) * numpy.pi * (d * 1e-6) ** 3 * Pdist
        return (Volume.sum() * m ** 3).asUnit(ml),0*ml


    def _Cough_Duguid(self):
        """
            From Duguid 1947.

            See data/duguio.ipynb
        :return:
        """

        smallDropletCough = 5.508527e-04*ml
        largeDropletCough = 0.059860*ml

        return smallDropletCough,largeDropletCough

    def _Sneeze_Duguid(self):
        """
            From Duguid 1947.

            See data/duguio.ipynb
        :return:
        """
        smallDropletCough = 3.862652e-02*ml
        largeDropletCough = 4.356433*ml

        return smallDropletCough, largeDropletCough

    def _Talk_Duguid(self):
        """
            From Duguid 1947.

        :return:
        """
        smallDropletCough = 2.998518e-05*ml
        largeDropletCough = 0.002579*ml

        return smallDropletCough, largeDropletCough
