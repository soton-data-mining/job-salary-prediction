#!/usr/bin/env python
from models.OldMainMethod import OldMainMethod
from models.SimpleAverage import SimpleAverage
from models.StandaloneSimilarity import StandaloneSimilarity

if __name__ == "__main__":

    # as each Model gets it's own copy of the data due to train_test_split(), we want to
    # free up this copy as we are going to potentially run multiple models here
    # a reference to the class in the scope will prevent GC

    old_main = OldMainMethod(load_location=True)
    old_main.run()
    del old_main

    similarity = StandaloneSimilarity(test_size=100, train_size=300)
    similarity.run()
    del similarity

    avg = SimpleAverage()
    avg.run()
    del avg
