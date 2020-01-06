# -*- coding: utf-8 -*-

import numpy as np

def dehaze(hazy, transmission_map, atmospheric_light):
    transmission_map += 1e-5
    dehazy = (hazy - atmospheric_light) / transmission_map + atmospheric_light

    return dehazy

def haze(dehazy, transmission_map, atmospheric_light):
    hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)

    return hazy

data_range = np.arange(-1.0, 1.0, 0.1)
tran_range = np.arange(0.0, 1.0, 0.1)

print('Dehaze...')
for hazy in data_range:
    for transmission_map in tran_range:
        for atmospheric_light in data_range:
            dehazy = dehaze(hazy, transmission_map, atmospheric_light)
            if dehazy > 1.0 or dehazy < -1.0:
                print('[Wrong]\tdehazy = {} (hazy = {}, transmission map = {}, atmospheric light = {})'.format(dehazy, hazy, transmission_map, atmospheric_light))

print('Haze...')
for dehazy in data_range:
    for transmission_map in tran_range:
        for atmospheric_light in data_range:
            hazy = haze(dehazy, transmission_map, atmospheric_light)
            if hazy > 1.0 or hazy < -1.0:
                print('[Wrong]\thazy = {} (dehazy = {}, transmission map = {}, atmospheric light = {})'.format(hazy, dehazy, transmission_map, atmospheric_light))