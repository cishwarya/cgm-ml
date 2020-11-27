USE_HEIGHT = True
USE_WEIGHT = False

assert USE_HEIGHT != USE_WEIGHT, "Please enable either HEIGHT or WEIGHT"

if USE_HEIGHT:
    from .qa_config_height import *
elif USE_WEIGHT:
    from .qa_config_weight import *
else:
    raise Exception("Please choose either weight or height!")
