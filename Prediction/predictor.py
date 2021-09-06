import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def data_values(data):
    data_field_types = {'address': 'object',
                        'postal_code': 'object',
                        'floor': 'float64',
                        'heating': 'object',
                        'homeType': 'object',
                        'numBathrooms': 'float64',
                        'numBedrooms': 'float64',
                        'parking': 'float64',
                        'price': 'float64',
                        'pricePerSqm': 'float64',
                        'area': 'float64',
                        'year': 'float64',
                        'LivingRooms': 'float64',
                        'Kitchens': 'float64',
                        'WC': 'float64',
                        'Orientation': 'object',
                        'NewlyBuilt': 'float64',
                        'Storage': 'float64',
                        'Views': 'float64',
                        'RoofTop': 'float64',
                        'SwimmingPool': 'float64',
                        'RoadFront': 'float64',
                        'Corner': 'float64',
                        'Renovated': 'float64',
                        'YearRenovated': 'float64',
                        'NeedsRenovation': 'float64',
                        'ListedBuilding': 'float64',
                        'Neoclassico': 'float64',
                        'Unfinished': 'float64',
                        'ProfessionalUse': 'float64',
                        'DistanceFromSea': 'float64',
                        'EnergyCat': 'object',
                        'Luxury': 'float64',
                        'StudentAccomodation': 'float64',
                        'SummerHouse': 'float64',
                        'BuildingZone': 'object',
                        'category': 'float64',
                        'lat': 'float64',
                        'lng': 'float64',
                        'place_id': 'object',
                        'numWCBath': 'float64',
                        'numRooms': 'float64',
                        'areaPerRoom': 'float64',
                        'areaPerWCBath': 'float64',
                        'posFeatures': 'float64',
                        'negFeatures': 'float64',
                        }

    for (field, fieldType) in data_field_types.items():
        if fieldType == 'float64':
            data[field] = data[field].astype(float)
    return data


def select(data, y_name, x_names):
    x_set = pd.DataFrame()
    for field in x_names:
        x_set[field] = data[field]
    y_set = data[y_name]

    return y_set, x_set


# standardization scaler - fit&transform on train, fit only on test

def scale_data(x_train_raw, x_test_raw, x_all_raw, x_no_trim_raw):
    s_scaler = StandardScaler()
    x_train_scaled = s_scaler.fit_transform(x_train_raw)
    x_test_scaled = s_scaler.transform(x_test_raw)
    if len(x_all_raw) > 0:
        x_all_scaled = s_scaler.transform(x_all_raw)
    else:
        x_all_scaled = []
    if len(x_no_trim_raw) > 0:
        x_no_trim_scaled = s_scaler.transform(x_no_trim_raw)
    else:
        x_no_trim_scaled = []
    return x_train_scaled, x_test_scaled, x_all_scaled, x_no_trim_scaled


# Multiple Liner Regression

def linear_regression(x_train, x_test, y_train, y_test, x_vars, verbose=False, graphs=False):
    x_train, x_test, _, _ = scale_data(x_train, x_test, [], [])

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # evaluate the model (intercept and slope)
    if verbose:
        print('Intercept: ', regressor.intercept_)

    # predicting the test set result

    y_pred = regressor.predict(x_test)

    # put results as a DataFrame

    coeff_df = pd.DataFrame(regressor.coef_, x_vars, columns=['Coefficient'])
    if verbose:
        print(coeff_df)

    # visualizing residuals
    if graphs:
        residuals = (y_test - y_pred)
        sns.displot(residuals)

    # compare actual output values with predicted values

    dfs = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = dfs.head(25)
    if verbose:
        print(df1)

    # evaluate the performance of the algorithm (MAE - MSE - RMSE - R^2)

    # MAE is particularly interesting because it is robust to outliers.
    # The loss is calculated by taking the median of all absolute differences
    # between the target and the prediction.  1/n Sum(|y-y_pred|)
    MAE = metrics.mean_absolute_error(y_test, y_pred)

    # a risk metric corresponding to the expected value of the squared (quadratic) error or loss.
    # 1/n Sum[(y-y_pred)^2]
    MSE = metrics.mean_squared_error(y_test, y_pred)

    # square root of MSE
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    #  If y_pred is the estimated target output, y the corresponding (correct) target output, and
    #  is Variance, the square of the standard deviation, then the explained variance is estimated
    #  as follow:  1- VAR(y-y_pred) / VAR(y)
    VarScore = metrics.explained_variance_score(y_test, y_pred)

    # it represents the proportion of variance (of y) that has been explained by the independent
    # variables in the model. It provides an indication of goodness of fit and therefore a measure
    # of how well unseen samples are likely to be predicted by the model, through the proportion
    # of explained variance.
    R2Score = metrics.r2_score(y_test, y_pred)

    if verbose:
        print("MAE: ", MAE)
        print("MSE: ", MAE)
        print("RMSE: ", MAE)
        print("VarScore: ", VarScore)
        print("R2: ", MAE)

    return MAE, MSE, RMSE, VarScore, R2Score, coeff_df, dfs, regressor.intercept_


# run a single or multi-variable linear regression for those X-vars


def multi_var_regression(yx_list, lin_reg_results, data, home_name='All', cat_name='All',
                         verbose=True, var_threshold=0.05, graphs=False):
    for yx in yx_list:
        y_name = yx[0]
        x_name = "{} ".format(yx[1]).strip('[] ').strip('\'')
        y, x = select(data, y_name, yx[1])

        # splitting Train and Test
        x_train, x_test, y_train, y_test = train_test_split(x.values.tolist(),
                                                            y.values.tolist(),
                                                            shuffle=True,  # shuffle to mix addresses
                                                            test_size=0.33,
                                                            random_state=101)

        MAE, MSE, RMSE, VarScore, R2Score, coeff_df, dfs, intercept = linear_regression(
            x_train, x_test, y_train, y_test, yx[1], verbose=False, graphs=graphs)
        output = {'y': y_name, 'x_vars': x_name, 'MAE': MAE, 'MSE': MSE,
                  'RMSE': RMSE, 'VarScore': VarScore, 'R2Score': R2Score}

        if VarScore >= var_threshold:
            output['pick'] = 1
        else:
            output['pick'] = 0
        output['category'] = cat_name
        output['homeType'] = home_name
        lin_reg_results = lin_reg_results.append(output, ignore_index=True)
        if verbose:
            print(lin_reg_results)

    return lin_reg_results


# Run regression analysis
###############################################################################################################


def regress(results):
    # run a multi-variable linear regression for those X-vars that have R-square >= 0.05

    yx_pairings = []
    for X in X_all:
        for yName in y_vars:
            x_select = results[results['y'] == yName]
            yx_pairings.append((yName, [X]))  # list with y-var key vs single X variable
    results = multi_var_regression(yx_pairings, results, df, var_threshold=0.05, verbose=True, graphs=False)

    # run a multi-variable linear regression for those X-vars that have R-square >= 0.05
    # from linear regression data results

    yx_pairings = []
    for y in y_vars:
        x_select = results[results['y'] == y]
        x_select = x_select[x_select['pick'] == 1]
        x_vars = x_select['x_vars'].tolist()
        yx_pairings.append((y, x_vars))  # collect the significant x-var names into dictiorary with y-var key

    results = multi_var_regression(yx_pairings, results, df, var_threshold=0.05, verbose=True, graphs=True)

    # run a multi-variable linear regression for by Category and Home Type
    # for those X-vars that have R-square >= 0.05

    for cat in range(2):
        cat_data = df[df['category'] == cat + 1]
        results = multi_var_regression(yx_pairings, results, cat_data, cat_name=str(cat + 1),
                                       var_threshold=0.05, verbose=True, graphs=True)

    house_list = ['Bungalow', 'Villa', 'Building', 'Mezonet', 'House']
    apartment_list = ['Studio', 'Apartment', 'BlockFlats', 'Loft']

    home_vals = house_list + apartment_list

    for i, homeType in enumerate(home_vals):
        home_data = df[df['homeType'] == homeType]
        if homeType in apartment_list:
            cat_name = '1'
        else:
            cat_name = '2'
        if len(home_data) > 30:  # if sample too small skip regression
            results = multi_var_regression(yx_pairings, results, home_data, home_name=homeType, cat_name=cat_name,
                                           var_threshold=0.05, verbose=True, graphs=False)
    return results


#  Test various machine learning optimization algorithms
########################################################


def get_optimizer(opt, lr):
    if opt == "SGD":
        return optimizers.SGD(lr)
    elif opt == "Adadelta":
        return optimizers.Adadelta(lr)
    elif opt == "Adamax":
        return optimizers.Adamax(lr)
    elif opt == "Nadam":
        return optimizers.Nadam(lr)
    elif opt == "RMSprop":
        return optimizers.RMSprop(lr)
    elif opt == "Adam":
        return optimizers.Adam(lr)
    else:
        return optimizers.Optimizer(lr)


def run_model(batch_size=32, dropout_p=0.9, opt="SGD", lr=0.001, epochs=128):
    # having as many neurons the number of X independent variables (YX[1])
    # 4 hidden layers and 1 output layer to predict house price.
    # ADAM optimization algorithm is used for optimizing loss function (MSE)

    model = Sequential()
    model.add(Dense(len(YX[1]), activation='relu'))
    model.add(Dense(len(YX[1]), activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(len(YX[1]), activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(len(YX[1]), activation='relu'))
    model.add(Dense(1))

    # train the model for epochs, and each time record the training and validation accuracy in the history object.
    # To keep track of how well the model is performing for each epoch, the model will run in both train and test data
    # along with calculating the loss function

    model.compile(optimizer=get_optimizer(opt, lr), loss='MSE')
    hist = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),
                     verbose=1, batch_size=batch_size, epochs=epochs)
    model.summary()

    # MODEL EVALUATION

    scor = model.evaluate(X_test, y_test, verbose=0)
    return hist, scor, model


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    filename = "../Analysis/FinalDataSet.txt"

    df = pd.read_csv(filename, dtype=str, keep_default_na=False, encoding='utf8',
                     delimiter='~', index_col=False, error_bad_lines=True)
    df.info()
    df = data_values(df)
    df.info()

    df_NoTrim = pd.read_csv("../Analysis/FinalDataSetNoTrim.txt", dtype=str, keep_default_na=False, encoding='utf8',
                            delimiter='~', index_col=False, error_bad_lines=True)

    df_NoTrim.info()
    df_NoTrim = data_values(df_NoTrim)
    df_NoTrim.info()

    # Run through all independent variables to test relevance to linear regression prediction of dependent variable

    y_vars = ['price', 'pricePerSqm']

    # Run main x variable (exclude 'numRooms', 'numWCBath' but use 'areaPerRoom', 'areaPerWCBath')
    #  'lat', 'lng'

    X_prime = ['floor', 'area', 'year', 'category', 'lat', 'lng', 'areaPerRoom', 'numWCBath',
               'numRooms', 'posFeatures', 'negFeatures']
    X_all = ['numBathrooms', 'numBedrooms', 'parking', 'area', 'year', 'LivingRooms', 'Kitchens', 'WC', 'NewlyBuilt',
             'Storage', 'Views', 'RoofTop', 'SwimmingPool', 'RoadFront', 'Corner', 'Renovated', 'NeedsRenovation',
             'ListedBuilding', 'Neoclassico', 'Unfinished', 'ProfessionalUse', 'Luxury', 'StudentAccomodation',
             'SummerHouse', 'category', 'lat', 'lng', 'areaPerRoom', 'numWCBath', 'numRooms',
             'posFeatures', 'negFeatures']

    X_reduced = ['parking', 'area', 'year', 'NewlyBuilt', 'Storage', 'Views', 'RoofTop', 'SwimmingPool', 'RoadFront',
                 'Corner', 'Renovated', 'NeedsRenovation', 'ListedBuilding', 'Neoclassico', 'Unfinished',
                 'ProfessionalUse',
                 'Luxury', 'StudentAccomodation', 'SummerHouse', 'category', 'lat', 'lng', 'areaPerRoom', 'numWCBath',
                 'numRooms', 'posFeatures', 'negFeatures']

    results = pd.DataFrame(
        columns=['pick', 'category', 'homeType', 'y', 'x_vars', 'MAE', 'MSE', 'RMSE', 'VarScore', 'R2Score']
    )

    #  Linear Regressions
    ################################################################################
    results = regress(results)

    #  Keras Regressions
    ################################################################################

    # Creating a Deep Learning Model run for both 'price' and 'pricePerSqm' independent and X_all vars

    X_test1 = ['numBedrooms', 'parking', 'area', 'Storage', 'SwimmingPool', 'Luxury', 'category', 'areaPerRoom',
               'posFeatures', 'lat', 'lng']
    X_test2 = ['parking', 'area', 'year', 'Storage', 'SwimmingPool', 'Luxury', 'posFeatures', 'lat', 'lng']

    YX_pairings = [('price', X_reduced), ('pricePerSqm', X_reduced)]
    # , ('price', X_prime), ('pricePerSqm', X_prime),
    # ('price', X_test1), ('pricePerSqm', X_test1), ('price', X_test2), ('pricePerSqm', X_test2)

    modl = None
    for YX in YX_pairings:

        #  Prepare Data Sets

        Y = YX[0]
        X = "{} ".format(YX[1]).strip('[] ').strip('\'')
        # splitting Train and Test
        y, x = select(df, YX[0], YX[1])

        # data before trimming outliers
        y_NoTrim, x_NoTrim = select(df_NoTrim, YX[0], YX[1])

        print('y=', Y, y)
        print('x=', YX[1], x)
        X_train, X_test, y_train, y_test = train_test_split(x.values,
                                                            y.values,
                                                            shuffle=True,  # shuffle to mix addresses
                                                            test_size=0.33,
                                                            random_state=101)
        X_train, X_test, x, x_NoTrim = scale_data(X_train, X_test, x, x_NoTrim)

        # Explore Alternative Machine Learning Model Parameters and Models

        batch_size_test_values = [32]  # [64, 32, 16]
        epochs_test_values = [512]  # [256, 128, 64]
        lr_test_values = [0.01]  # [0.001, 0.01, 0.1]
        Dropout_p_test_values = [0.05]  # [0.05, 0.50]
        optimizer_list = ["Adamax"]  # ["Adam", "Adadelta", "Adamax", "Nadam", "RMSprop"]
        out = list()
        colors = ['g', 'b', 'y', 'r', 'c', 'k']

        for batch_size in iter(batch_size_test_values):
            for lr in iter(lr_test_values):
                for Dropout_p in iter(Dropout_p_test_values):
                    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(10, 16))
                    runTitle = "LR_" + str(lr) + "_BS_" + str(batch_size) + "_Dp_" + str(Dropout_p)
                    plt.title(runTitle, fontsize=12, fontweight='bold')
                    ci = 0
                    for optimizer in iter(optimizer_list):
                        for epochs in iter(epochs_test_values):
                            history, score, modl = run_model(batch_size, Dropout_p, optimizer, lr, epochs)
                            out.append([optimizer, batch_size, lr, Dropout_p, epochs, score])
                            print('Model Opt:', optimizer, 'Batch Size:', batch_size, 'Learn Rate:', lr, 'Dropout p:',
                                  Dropout_p, 'Epochs:', epochs, 'Test loss:', score)
                            label_detail = "_epchs_" + str(epochs) + "_mdl-" + optimizer
                            ax1.plot(history.history['loss'], colors[ci], label="train " + label_detail)
                            ax2.plot(history.history['val_loss'], colors[ci], label="test " + label_detail)
                            ax1.grid(True)
                            ax2.grid(True)
                            plt.subplot(ax1)
                            plt.legend(loc='lower left')
                            plt.ylabel("Train MSE")
                            plt.subplot(ax2)
                            plt.legend(loc='lower left')
                            plt.ylabel("Test MSE")
                            plt.xlabel('Epochs', {'color': 'black', 'fontsize': 12})
                            plt.savefig(Y + runTitle + ".png")

                            y_pred = modl.predict(X_test)

                            MAE = metrics.mean_absolute_error(y_test, y_pred)
                            MSE = metrics.mean_squared_error(y_test, y_pred)
                            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            VarScore = metrics.explained_variance_score(y_test, y_pred)
                            R2Score = metrics.r2_score(y_test, y_pred)

                            print('MAE:', MAE)
                            print('MSE:', MSE)
                            print('RMSE:', RMSE)
                            print('VarScore:', VarScore)
                            print('R2Score:', R2Score)

                            if VarScore > 0.05:
                                pick = 1
                            else:
                                pick = 0

                            output = {'pick': pick, 'category': runTitle, 'homeType': label_detail, 'y': Y, 'x_vars': X,
                                      'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE, 'VarScore': VarScore, 'R2Score': R2Score}

                            results = results.append(output, ignore_index=True)
                        ci += 1

        # Calculate predictions for whole dataset without trimming the outliers

        df[Y + "_P"] = modl.predict(x)
        residuals = df[Y] - df[Y + "_P"]
        df[Y + '_StdErr'] = residuals / np.sqrt(metrics.mean_squared_error(df[Y], df[Y + "_P"]))
        sns.displot(residuals)
        plt.savefig("Residuals3Predictions{}.png".format(Y))
        sns.displot(df[Y + '_StdErr'])
        plt.savefig("Residuals3StdErr{}.png".format(Y))

        df_NoTrim[Y + "_P"] = modl.predict(x_NoTrim)
        residuals = df_NoTrim[Y] - df_NoTrim[Y + "_P"]
        df_NoTrim[Y + '_StdErr'] = residuals / np.sqrt(metrics.mean_squared_error(df_NoTrim[Y], df_NoTrim[Y + "_P"]))
        sns.displot(residuals)
        plt.savefig("Residuals3Predictions_NoTrim{}.png".format(Y))
        sns.displot(df_NoTrim[Y + '_StdErr'])
        plt.savefig("Residuals3StdErr_NoTrim{}.png".format(Y))

    results.to_csv("KerasAdamax516_32_01.txt", index=False, sep='~', encoding='utf-8')
    print(len(df))
    print(len(df_NoTrim))
    df.to_csv("REListingsPredictions516_32_01.txt", index=False, sep='~', encoding='utf-8')
    df_NoTrim.to_csv("REListingsPredictions516_32_01_NoTrim.txt", index=False, sep='~', encoding='utf-8')

    conn = sqlite3.connect('../Server/REListings.db')

    result = df.to_sql("RE_Listing", conn, if_exists='replace', index=True, index_label="id")

    # plt.show()
