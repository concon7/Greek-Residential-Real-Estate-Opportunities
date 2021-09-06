import urllib

import numpy as np
import pandas as pd
import requests


# =============================================================
# Find the geolocations from formatted_address field in database
# =============================================================

def find_geo_locations(addresses):
    num_addresses = len(addresses)
    rows = pd.DataFrame()  # each row corresponds to one address
    i = 0
    lat_athens = 37.9817858  # Athens Latitude
    lng_athens = 23.7430565  # Athens Longitude

    for address in addresses['address']:

        i = i + 1
        print('Getting region #%d of %d: %s' % (i, num_addresses, address.encode('utf8')))
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address=%s&key=AIzaSyBT2r4QfJtay3na4uoM3nyLoQGUMuTz8-c' \
              % urllib.parse.quote(address)
        try:
            json = requests.get(url).json()
        except:
            continue

        if len(json['results']) > 1:
            print(">1 address returned by google: ", json['results'][0]['formatted_address'])
            normalized = pd.json_normalize(json['results'][0])
        else:
            normalized = pd.json_normalize(json['results'])

        row = pd.DataFrame.from_dict(normalized, orient='columns')  # store in row json info

        if len(row) > 0:
            # if geolocation info is '' set to 0
            # get latitude
            row['geometry.location.lat'] = np.where(row['geometry.location.lat'] == '', 0.0,
                                                    row['geometry.location.lat'].astype(float))
            # get longitude
            row['geometry.location.lng'] = np.where(row['geometry.location.lng'] == '', 0.0,
                                                    row['geometry.location.lng'].astype(float))
            row.insert(0, 'address', address)  # put address we used in front of the json data
            rows = rows.append(row, sort=False)  # add each row of json data to the rows table

    address_lat_lng = pd.DataFrame()
    address_lat_lng['address'] = rows['address']
    address_lat_lng['lat'] = rows['geometry.location.lat']
    address_lat_lng['lng'] = rows['geometry.location.lng']
    address_lat_lng['place_id'] = rows['place_id']
    return address_lat_lng


# Calculate distance between two geolocations
##############################################

def distance(lat1, lng1, lat2, lng2):
    dist = np.arccos(
        np.sin(lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180) +
        np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) *
        np.cos(lng2 * np.pi / 180 - lng1 * np.pi / 180)
    ) * 6371

    return dist


# ======
# Main
# ======
if __name__ == '__main__':
    filename = "NeighbourhoodTown.txt"

    df = pd.read_csv(filename, dtype=str, keep_default_na=False, encoding='utf8',
                     delimiter='\t', index_col=False, error_bad_lines=True)

    newGeoFields = find_geo_locations(df)

    indexFilename = '%s-geolocs.txt' % filename.replace('.txt', '')
    newGeoFields.to_csv(indexFilename, index=False, sep='~', encoding='utf-8')
