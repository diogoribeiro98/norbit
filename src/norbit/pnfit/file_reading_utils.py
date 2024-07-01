import numpy as np

def get_line_index(line_to_match, fname):
    """ Given a string and a filename, searches for lines that match it.
        If more than one line is found, an error is raised.
    Args:
        line_to_match (string): line to search for in the file
        fname         (string): name of input file

    Returns:
        int : number of first line that matches the input string
    """

    with open(fname, "r") as fp:

        #Search file
        idx_list = [idx for idx,line in enumerate(fp) if line == line_to_match]
        
        if len(idx_list) > 1:

            error_message = "ERROR:"
            error_message += " more than one line matching \n \n"
            error_message += "\t {} \n \n".format(line_to_match) 
            error_message += " was found in file {}".format(fname)
            
            raise ValueError(error_message)

        elif idx_list == []:
            return None
        
        return idx_list[0]

def readlines_from_to(fname, n1, n2):
    """Reads file from lines n1 to n2

    Args:
        fname (string): input file to read
        n1    (int): first line to read
        n2    (int): last line to read

    Returns:
        array : array with lines
    """
    
    output = []

    with open(fname, "r") as fp:
        full_data = fp.readlines()
    
    data = full_data[n1:n2]

    for line in data:            
        sline = line.partition(';')
        if sline[0] != '':
            output.append(np.array(sline[0].split())[:].astype(float))

    return np.array(output)
       
