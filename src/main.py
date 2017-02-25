#!/usr/bin/env python
from models.OldMainMethod import OldMainMethod
from models.StandaloneSimilarity import StandaloneSimilarity


if __name__ == "__main__":
    old_main = OldMainMethod(load_location=True)
    old_main.run()

    similarity = StandaloneSimilarity()
    similarity.run()
