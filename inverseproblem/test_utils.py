

def its_a_vector_field(E):
    return E.shape[1] == 3


def its_a_scalar_field(E):
    """ Assert there is no dimensionality, just scalars """
    return len(E.shape) == 1


def we_have_field_at_all_points(points, efield):
    return len(efield) == len(points)


