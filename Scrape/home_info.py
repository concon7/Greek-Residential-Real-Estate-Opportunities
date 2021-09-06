from datetime import datetime


# ============================================
# Class to hold home info from spitogatos.gr
# ============================================
class HomeInfo:
    def __init__(
            self,
            url='',
            runDate='',
            price='',
            pricePerSqm='',
            sqm='',
            address='',
            homeType='',
            buildingCoefficient='',
            heating='',
            numBedrooms='',
            numBathrooms='',
            floor='',
            parking='',
            year='',
            code='',
            availableDate='',
            lastUpdate='',
            generalRegion='',
            specificRegion='',
            internalFeatures='',
            externalFeatures='',
            additionalFeatures=''
    ):
        # set all fields initially to ''

        for field in locals():
            setattr(self, field, '')

        # set date/time we downloaded the data

        self.runDate = datetime.now().strftime('%Y-%m-%d')

        # standardize the field names from those used in the website either Greek/english

        self.fieldLabelsEn = {
            'Price': 'price',
            'Price per m²': 'pricePerSqm',
            'Area': 'sqm',
            'Address': 'address',
            'Type': 'homeType',
            'Building coefficient': 'buildingCoefficient',
            'Heating System': 'heating',
            'Bedrooms': 'numBedrooms',
            'Bathrooms': 'numBathrooms',
            'Floor': 'floor',
            'Parking spot': 'parking',
            'Construction year': 'year',
            'Listing code': 'code',
            'Available since': 'availableDate',
            'Modified on': 'lastUpdate',
            'Neighborhood': 'specificRegion',
            'Internal': 'internalFeatures',
            'External': 'externalFeatures',
            'Extra': 'additionalFeatures'
        }

    # overriding the string representation methods to print the fields
    # that are not inbuilt functions or labels

    def __str__(self):
        fields = [
            field for field in dir(self) if
            not field.startswith('__') and
            field != 'fieldLabelsGr' and
            field != 'fieldLabelsEn' and
            field != 'self'
        ]

        # add all field names to a string separated by '~' and return
        # used to create the string to store in each line of the storage file

        return_str = ''
        for field in fields:
            return_str += getattr(self, field)
            return_str += '~'
        return return_str
