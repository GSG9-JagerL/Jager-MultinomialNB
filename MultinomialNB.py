#  Copyright (c) 2021 Jäger. All rights reserved.
import numpy as np
class MultinomialNB:
    def __init__(self)->None:
        pass
    def fit(self, X_train, y_train):
        class_unique = np.array([np.unique(y_train), np.arange(len(np.unique(y_)))])
        p_Y_alone = np.zeros(len(class_unique))
        for i in range(p_Y_alone):
            p_Y_alone

