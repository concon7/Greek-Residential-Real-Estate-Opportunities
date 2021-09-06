import numpy as np
import pandas as pd


# ==========================================================================================
# Clean the database fields and estimate relevance of each field by simple linear regression
# ==========================================================================================
# LIST OF DATABASE FIELD NAMES
# ------------------------------------------------------------------------------------------
# address Views	Unfinished
# postal_code area RoofTop ProfessionalUse
# floor	year SwimmingPool DistanceFromSea
# heating LivingRooms RoadFront	EnergyCat
# homeType Kitchens Corner Luxury
# numBathrooms WC Renovated StudentAccomodation
# numBedrooms YearRenovated	SummerHouse
# parking Orientation NeedsRenovation BuildingZone
# price	NewlyBuilt ListedBuilding
# pricePerSqm Storage Neoclassico
# ==========================================================================================


def cleanDatabase(df, geolocs):
    print('Number of listings: ', len(df), "Database Shape: ", df.shape)

    # Cleaning Tasks

    # Change address to 'Greece' for rows that have no address as we can ascertain their location

    print('Deleting %d listings with no address' % len(df[df['address'] == '']))
    df = df[df['address'] != '']

    # Delete all addresses that have no geolocation codes

    df['address'] = np.where(df.address.isin(geolocs['address']), df.address, 'delete')
    print('Deleting %d listings with no geolocations' % len(df[df['address'] == 'delete']))
    df = df[df['address'] != 'delete']

    df_orig_size = len(df)
    print('Number of listings: ', len(df), "Database Shape: ", df.shape)

    # Exclude listing that had a floor of 9 or more as an erroneous entry
    # Assign values to floor described by text

    floor_num_equival = {'Ημιόροφος': '0.5',  # intermediate ground and 1st floor
                         'Ημιυπόγειο': '-0.5',  # intermediate basement
                         'Ισόγειο': '0.0',  # ground floor
                         'Υπόγειο': '-1.0',  # basement/underground
                         '': '-2.0'}  # unknown floor

    print('listings at unkwown floor: %d from %d' % (len(np.where(df.floor == '')), df_orig_size))

    for floorName in floor_num_equival:
        df['floor'] = np.where(df.floor == floorName, floor_num_equival.get(floorName), df.floor)

    df['floor'] = df.floor.astype(float)  # Convert to float
    df = df[df['floor'] < 9]
    print("Ignoring %d listings with floor > 9 : " % (df_orig_size - len(df)))
    df_orig_size = len(df)

    # Exclude farms and non apartment/house listings such as Other Categories and Farm/Ranch
    # Create category field for 1: apartment 2: house

    house_list = ['Bungalow', 'Βίλλα', 'Κτίριο', 'Μεζονέτα', 'Μονοκατοικία']
    exclude_list = ['Φάρμα / Ράντσο', 'Λοιπές Κατηγορίες Κατοικίας']
    apartment_list = ['Studio / Γκαρσονιέρα', 'Διαμέρισμα', 'Συγκρότημα διαμερισμάτων', 'Loft']

    df['category'] = 1  # set all to 1
    df['category'] = np.where(df.homeType.isin(house_list), 2, 1)  # 2: house, 1: for apartment
    df = df[df.homeType.isin(exclude_list) == False]
    print("Excluded %d listings Not House or apartment : " % (df_orig_size - len(df)))
    print("Found %d apartment and %d houses" % (len(df[df['category'] == 1]), len(df[df['category'] == 2])))

    df_orig_size = len(df)

    # if number of bathrooms, living rooms, kitchens, WC, BuildingZone is blank put -1 for Unknown
    # and convert to float

    fields = ['LivingRooms', 'Kitchens', 'WC', 'numBathrooms',
              'numBedrooms', 'BuildingZone', 'DistanceFromSea'
              ]
    for field in fields:
        df[field] = np.where(df[field] == '', '-1', df[field])
        df[field] = df[field].astype(float)
        print("Found %d listings with %s number > 0 and %d with 0 vs %d undefined"
              % (len(df[df[field] > 0]), field, len(df[df[field] == 0]), len(df[df[field] < 0]))
              )

    # make pricePerSqm = price / area

    df['pricePerSqm'] = df.pricePerSqm.astype(float)
    df['price'] = df.price.astype(float)
    df['area'] = df.area.astype(float)
    df['pricePerSqm'] = df.price / df.area

    # Year of construction set to 0 if unknown
    # Year of construction set to -1 if under construction

    df['year'] = np.where(df.year == '-', '0', df.year)
    df['year'] = np.where(df.year == 'Υπό κατασκευή', '-1', df.year)
    df['year'] = df.year.astype(float)
    print("Found %d listings under construction" % len(df[df['year'] == -1]))

    # NewlyBuilt,Storage,Views,RoofTop,SwimmingPool,RoadFront,Corner,Renovated, YearRenovated fields set zero if blank
    fields = ['NewlyBuilt', 'Storage', 'Views', 'RoofTop',
              'SwimmingPool', 'RoadFront', 'Corner', 'Renovated',
              'YearRenovated', 'NeedsRenovation', 'ListedBuilding',
              'Neoclassico', 'ProfessionalUse', 'Luxury',
              'StudentAccomodation', 'SummerHouse', 'Unfinished'
              ]

    for field in fields:
        df[field] = np.where(df[field] == '', '0', df[field])
        df[field] = df[field].astype(float)
        print("Found %d listings with %s attribute checked" % (len(df[df[field] > 0]), field))

    # if year renovated < 2010 then set NeedsRenovation to 0

    df['Renovated'] = np.where(df.YearRenovated < 2010, 0, 1)
    print("Found %d listings with year renovated < 2010 of %d shown as renovated. Switched designation to not Renovated" %
          (len(df[df['YearRenovated'] > 0]) - len(df[df['YearRenovated'] >= 2010]), len(df[df['YearRenovated'] > 0]))
          )

    # if Unfinished=1 then NeedsRenovation=1 too

    df['NeedsRenovation'] = np.where(df.Unfinished == 1, 1, df.NeedsRenovation)
    print("Found %d listings under construction and ensured they are designated as needing renovation" %
          len(df[df['Unfinished'] == 1])
          )

    # if Distance from Sea < 500 meters set SummerHouse to 3
    # if Distance from 500 <= Sea < 2000 meters set new Variable proximitySea at 2
    # applies only to summer house, already '1' from above

    df['SummerHouse'] = np.where(df.DistanceFromSea < 2000, 2, df.SummerHouse)
    df['SummerHouse'] = np.where(df.DistanceFromSea < 500, 3, df.SummerHouse)
    print(
        "Assigned rating 3 to %d summer houses that are <500 meters and 2 to %d summer houses <2000 meters from sea and 1 "
        "to the %d rest. "
        % (len(df[df['SummerHouse'] == 3]), len(df[df['SummerHouse'] == 2]), len(df[df['SummerHouse'] == 1]))
    )
    # change parking designations from 'Οχι' (No) and 'Ναι' (yes) to 0 and 1.  Blanks should treated as 0

    df['parking'] = np.where(df.parking == 'Ναι', '1', '0').astype(float)
    print("Found %d listings with parking" % len(df[df['parking'] == 1]))

    # merge the geolocation dataframe with the data dataframe using the 'address field as an index
    # address field is unique

    geolocs['lat'] = geolocs['lat'].astype(float)
    geolocs['lng'] = geolocs['lng'].astype(float)

    df = pd.merge(df, geolocs, how='left', left_on='address', right_on='address')

    return df


# ======
# Main
# ======
if __name__ == '__main__':
    #  Run this program after running first findgeoloc.py to create the geolocations file

    dataFilename = "GreekRREs02Feb2019_ToClean.csv"  # data to clean up
    geolocsFilename = "NeighbourhoodTown-geolocs.txt"  # geolocations for each address
    outputFilename = "GreekRREs02Feb2019_Cleaned.txt"  # output cleaned up data file

    geolocs = pd.read_csv(geolocsFilename, dtype=str, keep_default_na=False, encoding='utf8',
                          delimiter='~', index_col=False, error_bad_lines=True)
    print('geolocs loaded')
    df = pd.read_csv(dataFilename, dtype=str, keep_default_na=False, encoding='utf8',
                     delimiter=',', index_col=False, error_bad_lines=False)
    print('df loaded')

    df = cleanDatabase(df, geolocs)

    df.to_csv(outputFilename, index=False, sep='~', encoding='utf-8')
