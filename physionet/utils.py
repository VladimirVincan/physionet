def create_mask(labels):
    mask = labels != -1
    labels = labels[mask]
    return mask
