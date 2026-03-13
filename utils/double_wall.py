def get_potential(x, H):
    # build the equation of the double wall (has the shape of w)
    return H * (x**4 - 2*x**2 + 1.0)