import datetime
import struct

def main():

    '''
    Since the idea of this project is to use hmms and neural network to predict stock prices,
    stock data is needed. In this case, we are going to use daily General eletric stock prices
    from 1962 to 2018.

    the raw data is in the file ge_us.txt, each of the 14058 lines is a string in the format:
    Date,Open,High,Low,Close,Volume,OpenInt. Since we only need the Date and closing prices, the data
    will be "refined" into binary structs of integers and floats, representing dates and prices

    ps: in order to kepp the data compact, the dates will not be storede directly, but in number of days
    passed from date_0 ( 1962 / 01 / 02 )
    '''

    form = "if"
    date_idx = 0
    closed_idx = 4

    f = open( "real_deal/ge_us.txt" , "r" )
    g = open( "real_deal/ge_comp.dat" , "ab" )

    cond = True
    it = 0 #counting the number of dates , prices
    f.readline() #header_line
    while cond:

        try:
            line = f.readline().split( sep = ",")
        except EOFError:
            cond = False
            continue

        #if above block fails
        cond = ( len( line ) > 1 )
        if not cond:
            continue

        year , month , day = ( int(x) for x in line[ date_idx ].split( "-" ) )
        if it == 0:
            date_0 = datetime.date( year , month , day )
        date = datetime.date( year , month , day )
        day_diff = ( date - date_0 ).days
        price = float( line[ closed_idx ] )
        print( day_diff , price )

        values = struct.pack( form , day_diff , price )
        g.write( values )

        it += 1
    
    f.close()
    g.close()

    print( date_0 )
    print( "number of days:", it )
    size = struct.calcsize( form )
    print( "size of pack:" , size , "bytes" )

    year , month , day = date_0.year , date_0.month , date_0.day
    metaform = "iiiii"
    meta_value = struct.pack( metaform , year , month , day , size , it )
    h = open( "real_deal/ge_meta.dat" , "wb" )
    h.write( meta_value )
    h.close()

if __name__ == "__main__":
    main()
