import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#  Create Violin Plots to Visualise Data Series by Variable Category and Type

def violin_plot(dataset, cat_name, cat_des, cat_vals, var_des, size):
    if len(cat_des) != len(cat_vals):
        print("Category Values and Descriptions need to be equal length")
        return
    fig, axs = plt.subplots(nrows=max(int((len(var_des) - 1) / 3) + 1, 1),
                            ncols=min(len(var_des), 3),
                            figsize=(size * min(len(var_des), 3),
                                     5 * max(int((len(var_des) - 1) / 3) + 1, 1)
                                     )
                            )
    for varType in range(len(var_des)):
        data = []
        for cat in range(len(cat_vals)):
            data_cat = dataset[dataset[cat_name] == cat_vals[cat]][var_des[varType]]
            if len(data_cat) > 0:
                data.append(data_cat)
            else:
                data.append([0])
        # adding horizontal grid lines
        if len(var_des) == 1:
            axs.violinplot(data,
                           showmeans=False,
                           showmedians=True)
            axs.set_title(var_des[varType])
            axs.yaxis.grid(True)
            axs.set_xticks([y + 1 for y in range(len(data))])
            axs.set_xlabel(cat_name)
            axs.set_ylabel('Observed values')
        elif len(var_des) <= 3:
            axs[varType].violinplot(data,
                                    showmeans=False,
                                    showmedians=True)
            axs[varType].set_title(var_des[varType])
            axs[varType].yaxis.grid(True)
            axs[varType].set_xticks([y + 1 for y in range(len(data))])
            axs[varType].set_xlabel(cat_name)
            axs[varType].set_ylabel('Observed values')
        else:
            axs[int(varType / 3)][varType % 3].violinplot(data,
                                                          showmeans=False,
                                                          showmedians=True)
            axs[int(varType / 3)][varType % 3].set_title(var_des[varType])
            axs[int(varType / 3)][varType % 3].yaxis.grid(True)
            axs[int(varType / 3)][varType % 3].set_xticks([y + 1 for y in range(len(data))])
            axs[int(varType / 3)][varType % 3].set_xlabel(cat_name)
            axs[int(varType / 3)][varType % 3].set_ylabel('Observed values')

    # add x-tick labels
    plt.setp(axs, xticks=[y + 1 for y in range(len(cat_vals))],
             xticklabels=cat_des)
    plt.tight_layout()
    return fig


#  return dataset between low and high standard deviations of varName


def reduce_to_std_range(data, var_name, low, high):
    mean = np.mean(data[var_name])
    std = np.std(data[var_name])
    low_range = mean + low * std
    high_range = mean + high * std
    data = data[data[var_name] > low_range]
    data = data[data[var_name] < high_range]
    return data


# Code from matplotlib for survey style data analysis
#####################################################


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('GnBu')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


#   Analyze and Improve Statistical Quality of Data
#   Run Linear Regression Analysis to Evaluate Important to Prediction Model
#   Select final Deep Learning Data Set
##############################################################################


def field_stats(dfs, trim_outliers):
    # set the field that want to statistically test

    dfs['price'] = dfs['price'].astype(float)
    dfs['pricePerSqm'] = dfs['pricePerSqm'].astype(float)

    x_vars = ['area', 'floor', 'year', 'lat', 'lng', 'category',
              'LivingRooms', 'Kitchens', 'numBathrooms', 'WC', 'numBedrooms',
              'Views', 'Unfinished', 'RoofTop', 'ProfessionalUse',
              'SwimmingPool', 'RoadFront', 'Corner', 'Luxury',
              'Renovated', 'StudentAccomodation', 'SummerHouse',
              'parking', 'NeedsRenovation', 'NewlyBuilt', 'ListedBuilding',
              'Storage', 'Neoclassico']

    for x in x_vars:
        print('converting to float ...', x)
        dfs[x] = dfs[x].astype(float)

    dfs.info()

    # reduce dataset to prices within 3 standard deviations of mean of pricePerSqm
    ###############################################################################

    if trim_outliers:
        dfs = reduce_to_std_range(dfs, 'pricePerSqm', -3, 3)

    # visualizing house prices in price vs. pricePerSqm

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2], ['price', 'pricePerSqm'], 5)
    fig.savefig("ViolinPricevsPPSqmDist6stds{}.png".format(trim_outliers))
    plt.close(fig)

    # distribution of listings by lat vs. lng by category
    ######################################################

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 1, 1)
    sns.displot(dfs, x="lng", y="lat", hue='category', color='br')
    plt.tight_layout()
    fig.add_subplot(2, 1, 2)
    sns.displot(dfs, x="lng", y="lat", kind="kde", hue='category', color='br')
    plt.tight_layout()

    plt.savefig("LatLngDist{}.png".format(trim_outliers))
    plt.close(fig)

    # visualizing area, floor, year, category vs price per sqm
    ##########################################################

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2], ['floor', 'year', 'area'], 5)
    plt.savefig("FloorAreaYear{}.png".format(trim_outliers))
    plt.close(fig)

    # reduce dataset to listings with areas within 3 standard deviations of mean of area
    # reduce dataset to listings with defined >-2 floor levels and defined > 1900 construction years
    ############################################################################################

    if trim_outliers:
        dfs = reduce_to_std_range(dfs, 'area', -3, 3)
        dfs = dfs[dfs['year'] > 1900]
    dfs = dfs[dfs['floor'] > -2]

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2], ['floor', 'year', 'area'], 5)
    print('Dataset reduced to:', len(dfs))
    plt.savefig("FloorAreaYearReduced{}.png".format(trim_outliers))
    plt.close(fig)

    # visualizing number bathrooms, WCs, bedrooms, kitchens, living rooms vs. price per sqm
    #######################################################################################

    dfs['numWCBath'] = np.where(dfs.numBathrooms == -1, 0, dfs.numBathrooms
                                ) + np.where(dfs.WC == -1, 0, dfs.WC)
    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2],
                      ['numBedrooms', 'LivingRooms', 'numBathrooms',
                       'Kitchens', 'WC', 'numWCBath'], 5)

    plt.savefig("PricePerSqmNumRooms{}.png".format(trim_outliers))
    plt.close(fig)

    # visualizing number bathrooms, WCs, bedrooms, kitchens, living rooms vs. price per sqm
    #######################################################################################

    dfs['numRooms'] = np.where(dfs.numBedrooms == -1, 0, dfs.numBedrooms
                               ) + np.where(dfs.LivingRooms == -1, 0, dfs.LivingRooms
                                            ) + np.where(dfs.Kitchens == -1, 0, dfs.Kitchens)
    if trim_outliers:
        dfs = dfs[dfs['numRooms'] > 0]
        print('Eliminating listings if numRooms = 0, reduced to:', len(dfs))

    # delete listings that have no rooms and area per room > 200 sqm

    dfs['areaPerRoom'] = np.where(dfs.numRooms > 0, dfs.area / dfs.numRooms, 0)
    dfs['areaPerWCBath'] = np.where(dfs.numWCBath > 0, dfs.area / dfs.numWCBath, 0)
    if trim_outliers:
        dfs = dfs[dfs['areaPerRoom'] <= 200]
        print('Eliminating listings if area per room > 200, reduced to:', len(dfs))

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2],
                      ['areaPerRoom', 'areaPerWCBath'], 5)

    plt.savefig("PricePerSqmNumRoomsDef{}.png".format(trim_outliers))
    plt.close(fig)

    # positive and negative features vs price per Sqm
    ##################################################

    dfs['posFeatures'] = dfs['Views'] + dfs['RoofTop'] + dfs['ProfessionalUse'] + dfs['SwimmingPool'] + dfs[
        'RoadFront'] + dfs['Corner'] + dfs['Luxury'] + dfs['Renovated'] + dfs['StudentAccomodation'] + dfs[
                             'SummerHouse'] + dfs['parking'] + dfs['NewlyBuilt'] + dfs['Neoclassico'] + dfs['Storage']

    dfs['negFeatures'] = dfs['ListedBuilding'] + dfs['Unfinished'] + dfs['NeedsRenovation']

    category_names = ['No(=0)', 'Yes(=1)']
    results = {}
    fields = ['Views', 'RoofTop', 'ProfessionalUse', 'SwimmingPool', 'RoadFront', 'Corner',
              'Luxury', 'Renovated', 'StudentAccomodation', 'parking', 'NewlyBuilt',
              'Neoclassico', 'Storage', 'ListedBuilding', 'Unfinished', 'NeedsRenovation']

    dfs_len = len(dfs)
    for field in fields:
        yes = np.sum(dfs[field])
        no = dfs_len - yes
        results.update({field: [no, yes]})

    fig, ax = survey(results, category_names)
    plt.tight_layout()

    plt.savefig("Features{}.png".format(trim_outliers))
    plt.close(fig)

    # Adding positive and negative features
    ################################################
    # Set up the axes with gridspec

    features = ['posFeatures', 'negFeatures']
    values = [15, 4]
    main = []

    for i in range(2):
        catDes = [str(x) for x in range(values[i])]
        catVals = [x for x in range(values[i])]
        fig = violin_plot(dfs, features[i], catDes, catVals, ['pricePerSqm'], 5)
        plt.tight_layout()
        plt.savefig("PricePerSqm{}{}.png".format(features[i], trim_outliers))
        plt.close(fig)

    #  Check more granular listing types price per square meter distributions
    ########################################################################

    houseList = ['Bungalow', 'Βίλλα', 'Κτίριο', 'Μεζονέτα', 'Μονοκατοικία']
    houseListDes = ['H:Bungalow', 'H:Villa', 'H:Building', 'H:Mezonet', 'H:House']
    apartmentList = ['Studio/Γκαρσονιέρα', 'Διαμέρισμα', 'Συγκρότημα διαμερισμάτων', 'Loft']
    apartmentDes = ['A:Studio', 'A:Apartment', 'A:BlockFlats', 'A:Loft']
    catVals = houseList + apartmentList
    catDes = houseListDes + apartmentDes

    fig = violin_plot(dfs, 'homeType', catDes, catVals, ['pricePerSqm'], 15)
    plt.tight_layout()
    plt.savefig("PricePerHomeType{}.png".format(trim_outliers))
    plt.close(fig)

    fig = violin_plot(dfs, 'homeType', catDes, catVals, ['areaPerRoom'], 15)
    plt.tight_layout()
    plt.savefig("AreaPerRoomPerHomeType{}.png".format(trim_outliers))
    plt.close(fig)

    # translate to english the home types in the dataset

    for i, house in enumerate(houseList):
        dfs['homeType'] = np.where(dfs['homeType'] == house, houseListDes[i].replace("H:", "", 1), dfs['homeType'])
    for i, apartment in enumerate(apartmentList):
        dfs['homeType'] = np.where(dfs['homeType'] == apartment, apartmentDes[i].replace("A:", "", 1), dfs['homeType'])

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2], ['pricePerSqm'], 5)
    plt.tight_layout()
    plt.savefig("FinalPricePerSqmPerHomeType{}.png".format(trim_outliers))
    plt.close(fig)

    fig = violin_plot(dfs, 'category', ['Apartment', 'House'], [1, 2], ['areaPerRoom'], 5)
    plt.tight_layout()
    plt.savefig("FinalAreaPerRoomPerHomeType{}.png".format(trim_outliers))
    plt.close(fig)

    return dfs


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    filename = "../Clean/GreekRREs02Feb2019_Cleaned.txt"

    df = pd.read_csv(filename, dtype=str, keep_default_na=False, encoding='utf8',
                     delimiter='~', index_col=False, error_bad_lines=True)

    df_result = field_stats(df, False)  # DON'T TRIM OUTLIERS, use for Django
    df_result.to_csv("FinalDataSetNoTrim.txt", index=False, sep='~', encoding='utf-8')

    df_result = field_stats(df, True)  # TRIM OUTLIERS, use for predictor.py
    df_result.info()

    df_result.to_csv("FinalDataSet.txt", index=False, sep='~', encoding='utf-8')
    # plt.show()
