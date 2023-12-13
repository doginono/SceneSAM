

def backproject(uv, T, T2, K):
    """map uv to uv from the other frame
        goal is to use broadcast to calculate uv2 for all samples in one go

    Args:
        uv (np.array): _description_
        T (np.array): includes both 
        K (np.array): _description_
        
    returns:
        uv2 (_type_): _description_
    """
    return None

def sample_from_instances(instances, points_per_instance):
    """samples uv from the instances

    Args:
        instances (numpy.array): numpy array with shape (height, width) and values in [0, len(masks)-1]
        
    returns:
        uv (numpy.array): shape(2,points_per_instance, len(instances))
    
    """
    pass