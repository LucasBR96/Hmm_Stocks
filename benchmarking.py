from trainer import trainer
from extractor import *

import matplotlib
import matplotlib.pyplot as plt
import numpy 

def benchmark( mod , X_train , y_train , X_test , y_test ):

    mod.fit( X_train , y_train )
    c1 = mod.cost( X_train , y_train ).cpu().numpy()
    c2 = mod.cost( X_test , y_test ).cpu().numpy()

    return c1 , c2

def hmm_benchmark( X_train , y_train , X_test , y_test , sizes ):

    m = [ min( sizes ) ]
    train_scores = [] 
    test_scores = []
    for s in sizes:
        mod = trainer( s , m )
        c1 , c2 = benchmark( mod , X_train , y_train , X_test , y_test )
        train_scores.append( c1 )
        test_scores.append( c2 )
    return train_scores , test_scores

def net_benchmark( X_train , y_train , X_test , y_test , sizes ):

    m = max( sizes ) 
    train_scores = [] 
    test_scores = []
    for s in sizes:
        mod = trainer( m , [ s ] )
        c1 , c2 = benchmark( mod , X_train , y_train , X_test , y_test )
        train_scores.append( c1 )
        test_scores.append( c2 )
    return train_scores , test_scores

def main():

    extract = data_extractor( "real_deal/ge_comp.dat" , 14058 , "if" )
    data = extract.sample_chunks( chunck_size = 41 )
    b_prop = get_prop( format_data( data , 1 ) )
    X , y = separate_Xy( b_prop )
    X_train , X_test = X[ :70 ] , X[ 70: ]
    y_train , y_test = y[ :70 ] , y[ 70: ]

    values = list( range( 2 , 11 ) )
    train_latent , test_latent = hmm_benchmark( X_train , y_train , X_test , y_test , values )
    train_net , test_net = net_benchmark( X_train , y_train , X_test , y_test , values )

    img , ( ax1 , ax2 ) = plt.subplots( 1 , 2 )
    img.suptitle('benchmarking of the pipeline')
    ax1.plot( numpy.array( values ) , numpy.array( train_latent )  , 'tab:blue' )
    ax1.plot( numpy.array( values ) , numpy.array( test_latent )  , 'tab:red' )
    ax2.plot( numpy.array( values ) , numpy.array( train_net)  , 'tab:blue' )
    ax2.plot( numpy.array( values ) , numpy.array( test_net )  , 'tab:red' )

    plt.show()

if __name__ == "__main__":
    main()

