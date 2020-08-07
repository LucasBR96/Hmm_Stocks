from trainer import trainer
from extractor import *

import matplotlib
import matplotlib.pyplot as plt
import numpy 

    
def main():

    '''
    After the benchmarking, here we are training the model for real
    '''
    OUTER_BATCH = 1000
    CHNK_SIZE = 101
    EXTRACTIONS = 225
    U_LIM = .7

    INNER_BATCH = 25
    LR = 1e-4
    NUM_ST = 8
    HIDDEN_L = [ 4 , 3 ]

    A = data_extractor( "real_deal/ge_comp.dat" , 14058 , "if" , upper_lim = U_LIM )
    mod = trainer( NUM_ST , HIDDEN_L )
    costs = []

    for i in range( EXTRACTIONS ):

        data = A.sample_chunks( chunck_size = CHNK_SIZE , num_chuncks = OUTER_BATCH )
        b_prop = get_prop( format_data( data , 1 ) )
        X , y = separate_Xy( b_prop )

        cost_data = mod.fit( X - 1 , y - 1 , iters = int( OUTER_BATCH/INNER_BATCH ) , batch_size = INNER_BATCH , lr = LR )
        costs.extend( cost_data )
    
    iters , cost_values = zip( *costs )
    plt.plot( numpy.array( iters ) , 10*numpy.array( cost_values ) , label = "costs")
    plt.legend()
    plt.show()

    torch.save( mod.mod.state_dict() , "real_deal/saved_model.dat" )

if __name__ == "__main__":
    main()