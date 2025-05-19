def create_mask_old(labels):
    mask = labels != -1
    labels = labels[mask]
    return mask

def create_mask(labels):
    mask = labels > -0.5
    labels = labels[mask]
    return mask


# TODO: add test to compare create_mask functions
# tested to work the same

        # print(mask.shape)
        # print(mask2.shape)

        # num_true = mask.sum()
        # num_false = (~mask).sum()
        # num_true05 = mask2.sum()
        # num_false05 = (~mask2).sum()
        # print(num_true)
        # print(num_false)
        # print(num_true05)
        # print(num_false05)
