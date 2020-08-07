from torch_mod import next_state_pipeline
from extractor import *

import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import datetime

def predict_price_seq( original_seq , chunk_size , mod ):

    price_0 = original_seq[ chunk_size - 1 ]
    values = [ price_0 ]
    norm = torch.distributions.Normal( 0 , 1 )

    n = len( original_seq ) - chunk_size
    for i in range( n ):

        seq = original_seq[ i : i + chunk_size ]
        var = torch.from_numpy( seq[ 1: ]/seq[ : -1 ] - 1 ).unsqueeze( 0 ).float()
        mu , sigma = mod( var )
        pred_var = ( mu + sigma*norm.sample_n( 1 ) ).squeeze().detach().numpy()

        new_value = seq[ -1 ]*( pred_var + 1)
        print( new_value )
        values.append( new_value )
    return numpy.array( values )

def select_dates( date_seq , start , date_0 ):

    dates = date_seq[ start - 1: ]
    formated = []
    for x in dates:
        date_i = date_0 + datetime.timedelta( days = int( x ) )
        formated.append( date_i )
    return numpy.array( formated )

def extract_data( ):

    extrc = data_extractor( "real_deal/ge_comp.dat" , 14058 , "if" , lower_lim = .7  )
    days , values = zip( *extrc.extract_all() )
    return numpy.array( days ) , numpy.array( values )

def plot_contents( days , real_values , predicted_values ):
    
    img , ( ax1 , ax2 ) = plt.subplots( 1 , 2 , sharex = True , sharey = True)
    img.suptitle('prices of GE stocks')

    ax1.plot( days , real_values , 'tab:red' )
    ax1.set_title( 'real values')
    ax2.plot( days , predicted_values )
    ax2.set_title( 'model')

    ax1.set( xlabel = 'days' , ylabel = "prices")
    

    # img.xlabel( "days" )
    # img.ylabel( "price" )
    
    plt.show()

def load_model( ):

    A = next_state_pipeline( 8 , [ 4 , 3 ] )
    A.load_state_dict( torch.load("real_deal/saved_model.dat") )
    A.eval()
    return A


def main():

    num = 100
    date_0 = datetime.date( 1962 , 1 , 2 )
    days , values = extract_data()
    mod = load_model()

    pred_val = predict_price_seq( values , num , mod )
    real_val = values[ num - 1: ]
    real_dates = select_dates( days , num , date_0 )

    # real_dates = numpy.arange( 10 )
    # real_val = numpy.cumsum( numpy.random.random( 10 ) )
    # pred_val = numpy.cumsum( numpy.random.random( 10 ) )

    plot_contents( real_dates , real_val , pred_val )

if __name__ == "__main__":
    main()



