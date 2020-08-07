import struct
import numpy
import torch

class data_extractor:

    def __init__( self , file_name , num_lines , form , lower_lim = 0 , upper_lim = 1 ):
        '''
        this class extracts blocks of sequential data from a binary file. Such file is made up
        of many structs of the same form, each one being a "line"
        '''

        self.addr = file_name
        self.f = open( file_name , "rb" ) #file object
        self.num_lines = num_lines #in this case a struct
        self.form = form #how each struct is formated

        #positinal boundries
        self.low = numpy.clip( lower_lim , 0 , .99 )
        self.high = numpy.clip( upper_lim , self.low , 1 )
    
    def validate_range( self , lower_value , upper_value = None ):

        min_block = numpy.floor( self.num_lines*self.low )
        max_block = numpy.ceil( self.num_lines*self.high )

        if upper_value is None:
            return min_block <= lower_value < max_block 
        else:
            return min_block <= lower_value < upper_value < max_block 
    
    def extract_chunck( self , pos , chunck_size ):

        if not self.validate_range( pos , pos + chunck_size ):
            raise RuntimeError

        if self.f.closed:
            self.f = open( self.addr , "rb" )
        
        start = pos*struct.calcsize( self.form )
        self.f.seek( start )
        byte_size = chunck_size*struct.calcsize( self.form )
        chunck = self.f.read( byte_size )
        iterator = struct.iter_unpack( self.form , chunck )

        return [ next( iterator ) for n in range( chunck_size ) ]

    def sample_chunks( self , num_chuncks = 100 , chunck_size = 10 ):

        min_block = numpy.floor( self.num_lines*self.low )
        max_block = numpy.ceil( self.num_lines*self.high ) - chunck_size
        idx_range = numpy.arange( min_block , max_block , dtype = int )
        idx = numpy.random.choice( idx_range , size = num_chuncks , replace = False )
        
        samples = [ self.extract_chunck( pos , chunck_size ) for pos in idx ]
        self.f.close()

        return samples
    
    def extract_all( self ):

        min_block = numpy.floor( self.num_lines*self.low )
        max_block = numpy.ceil( self.num_lines*self.high ) - 1
        return self.extract_chunck( int( min_block ) , int( max_block - min_block ) )

def format_data( sample , pos ):

    total_data = []
    for chunck in sample: # -> [ ( day_0 , price_0 ) , ( day_1 , price_1 ) , ...  ]
        row = [ pair[ pos ] for pair in chunck ]
        total_data.append( row )
    return torch.tensor( total_data )

def get_prop( data ):

    '''
    return row wise proportion between values
    in other words , for every row in data
    get row = ( x1 , x2 , x3 , x4 .....)
    returns ( x2/x1 , x3/x2 , ......)
    '''

    d_1 = data[ : , : -1 ]
    d_2 = data[ : , 1: ]
    return d_2/d_1

def separate_Xy( data ):

    return data[ : , :-1 ] , data[ : , -1 ]

# if __name__ == "__main__": 
#     A = data_extractor( "real_deal/ge_comp.dat" , 14058 , "if" )
#     b = A.sample_chunks( 5 , 10 )
    
#     b_prop = get_prop( format_data( b , 1 ) )
#     print( b_prop )