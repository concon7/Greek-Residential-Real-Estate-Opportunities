# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class ReListing(models.Model):
    id = models.IntegerField(primary_key=True)
    address = models.TextField(blank=True, null=True)
    postal_code = models.TextField(blank=True, null=True)
    floor = models.FloatField(blank=True, null=True)
    heating = models.TextField(blank=True, null=True)
    hometype = models.TextField(db_column='homeType', blank=True, null=True)  # Field name made lowercase.
    numbathrooms = models.FloatField(db_column='numBathrooms', blank=True, null=True)  # Field name made lowercase.
    numbedrooms = models.FloatField(db_column='numBedrooms', blank=True, null=True)  # Field name made lowercase.
    parking = models.FloatField(blank=True, null=True)
    price = models.FloatField(blank=True, null=True)
    pricepersqm = models.FloatField(db_column='pricePerSqm', blank=True, null=True)  # Field name made lowercase.
    area = models.FloatField(blank=True, null=True)
    year = models.FloatField(blank=True, null=True)
    livingrooms = models.FloatField(db_column='LivingRooms', blank=True, null=True)  # Field name made lowercase.
    kitchens = models.FloatField(db_column='Kitchens', blank=True, null=True)  # Field name made lowercase.
    wc = models.FloatField(db_column='WC', blank=True, null=True)  # Field name made lowercase.
    orientation = models.TextField(db_column='Orientation', blank=True, null=True)  # Field name made lowercase.
    newlybuilt = models.FloatField(db_column='NewlyBuilt', blank=True, null=True)  # Field name made lowercase.
    storage = models.FloatField(db_column='Storage', blank=True, null=True)  # Field name made lowercase.
    views = models.FloatField(db_column='Views', blank=True, null=True)  # Field name made lowercase.
    rooftop = models.FloatField(db_column='RoofTop', blank=True, null=True)  # Field name made lowercase.
    swimmingpool = models.FloatField(db_column='SwimmingPool', blank=True, null=True)  # Field name made lowercase.
    roadfront = models.FloatField(db_column='RoadFront', blank=True, null=True)  # Field name made lowercase.
    corner = models.FloatField(db_column='Corner', blank=True, null=True)  # Field name made lowercase.
    renovated = models.FloatField(db_column='Renovated', blank=True, null=True)  # Field name made lowercase.
    yearrenovated = models.FloatField(db_column='YearRenovated', blank=True, null=True)  # Field name made lowercase.
    needsrenovation = models.FloatField(db_column='NeedsRenovation', blank=True, null=True)  # Field name made lowercase.
    listedbuilding = models.FloatField(db_column='ListedBuilding', blank=True, null=True)  # Field name made lowercase.
    neoclassico = models.FloatField(db_column='Neoclassico', blank=True, null=True)  # Field name made lowercase.
    unfinished = models.FloatField(db_column='Unfinished', blank=True, null=True)  # Field name made lowercase.
    professionaluse = models.FloatField(db_column='ProfessionalUse', blank=True, null=True)  # Field name made lowercase.
    distancefromsea = models.FloatField(db_column='DistanceFromSea', blank=True, null=True)  # Field name made lowercase.
    energycat = models.TextField(db_column='EnergyCat', blank=True, null=True)  # Field name made lowercase.
    luxury = models.FloatField(db_column='Luxury', blank=True, null=True)  # Field name made lowercase.
    studentaccomodation = models.FloatField(db_column='StudentAccomodation', blank=True, null=True)  # Field name made lowercase.
    summerhouse = models.FloatField(db_column='SummerHouse', blank=True, null=True)  # Field name made lowercase.
    buildingzone = models.TextField(db_column='BuildingZone', blank=True, null=True)  # Field name made lowercase.
    category = models.FloatField(blank=True, null=True)
    lat = models.FloatField(blank=True, null=True)
    lng = models.FloatField(blank=True, null=True)
    place_id = models.TextField(blank=True, null=True)
    numwcbath = models.FloatField(db_column='numWCBath', blank=True, null=True)  # Field name made lowercase.
    numrooms = models.FloatField(db_column='numRooms', blank=True, null=True)  # Field name made lowercase.
    areaperroom = models.FloatField(db_column='areaPerRoom', blank=True, null=True)  # Field name made lowercase.
    areaperwcbath = models.FloatField(db_column='areaPerWCBath', blank=True, null=True)  # Field name made lowercase.
    posfeatures = models.FloatField(db_column='posFeatures', blank=True, null=True)  # Field name made lowercase.
    negfeatures = models.FloatField(db_column='negFeatures', blank=True, null=True)  # Field name made lowercase.
    price_p = models.FloatField(db_column='price_P', blank=True, null=True)  # Field name made lowercase.
    price_stderr = models.FloatField(db_column='price_StdErr', blank=True, null=True)  # Field name made lowercase.
    pricepersqm_p = models.FloatField(db_column='pricePerSqm_P', blank=True, null=True)  # Field name made lowercase.
    pricepersqm_stderr = models.FloatField(db_column='pricePerSqm_StdErr', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'RE_Listing'
