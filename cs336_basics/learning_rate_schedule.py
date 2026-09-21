import math

def learning_rate_schedule(t, amax, amin, tw, tc):
    if t < tw:
        return t * amax / tw
    if tw <= t <= tc:
        return amin + (1+math.cos((t-tw)*math.pi/(tc-tw)))*(amax-amin)/2
    if t > tc:
        return amin