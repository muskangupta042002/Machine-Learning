import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Funection to gt data
def get_data(file_name):
    dataframe = pd.read_csv(file_name)
    print( dataframe )
    x_parameters = []
    y_parameters = []
    for single_square_feet, single_price_value in zip(
            dataframe['square_feet'], dataframe['price'] ):

        x_parameters.append( [ single_square_feet] )
        y_parameters.append(  single_price_value  )

    return x_parameters,y_parameters

# Function for Fitting data to Linear model
def linear_model_main(X_parameters, Y_parameters, quest_value):
    # Create linear regression object
    regr = LinearRegression()
    print( "AREA : " , X_parameters  )
    print( "PRICE: " , Y_parameters  )
    regr.fit(X_parameters, Y_parameters)                  # m and c for y = mx + c   We are training the algorithm
    predicted_ans = regr.predict([ [quest_value] ])    #<--quest_value=700
    predictions = {}
    predictions['coefficient'] = regr.coef_      #  m or slope or regr.coef_
    predictions['intercept']    = regr.intercept_
    predictions['predicted_ans'] = predicted_ans

    print( "Output from Machine = " , predicted_ans  )

    plt.scatter(X_parameters, Y_parameters, color="m",
                                     marker="o", s=30)

    all_predicted_Y = regr.predict(X_parameters)

    plt.scatter(X_parameters, all_predicted_Y, color="b")

    plt.plot(X_parameters, all_predicted_Y, color="r")
    plt.scatter(quest_value, predicted_ans, color="g")
    plt.show()
    return predictions


#predicting house price for house of 700 square feet area
if __name__ == "__main__":
    x, y = get_data('LR_House_price.csv')

    question_value = 700  # This is the question
    result = linear_model_main(x,y,question_value)

    print( "coefficient m=" , result['coefficient']  )
    print( "Intercept value c=" , result['intercept']  )


    print( "Predicted House Price value: ", result['predicted_ans']  )

















#x,y = get_data('LR_House_price_temp.csv')








'''




#x,y = get_data('LR_House_price.csv')
#print( "x :  ", x
#print( "y :  ", y
#print( "\n\n************************\n\n"



def get_data(file_name):
    dataframe = pd.read_csv(file_name)
    print( dataframe
    array  = dataframe.values
    x_parameter = array[:, 0]
    y_parameter = array[:, 1]
    print( "values of x_parameter : ", x_parameter
    print( "values of y_parameter : ", y_parameter
    x_parameter = x_parameter.reshape(-1,1)
    return x_parameter ,y_parameter

#x,y = get_data('LR_House_price.csv')
#print( x
#print( y
'''