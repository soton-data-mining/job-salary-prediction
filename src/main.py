#!/usr/bin/env python
from models.OldMainMethod import OldMainMethod
from models.StandaloneSimilarity import StandaloneSimilarity

if __name__ == "__main__":
    old_main = OldMainMethod()
    old_main.run()

    similarity = StandaloneSimilarity(load_location=False)
    similarity.run()
