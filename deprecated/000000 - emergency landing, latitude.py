import numpy as np 
#cause why not
cmds={"emg_land_lat":0b000000,"emglandlong":0b000001,"qdest_fin_lat":0b000010,"qdest_fin_long":0b000011,"qloc_lat":0b000100,"qloc_long":0b000101,"q_airspd":0b000110,"qtype":0b000111,
  "qspdmax":0b001000,"r_freq_change":0b001001,"a_freq_change":0b001001,"dfreq_change":0b001011,"r_spd_change":0b001100,"a_spd_change":0b001101,"qscup":0b001110,"qsclow":0b001110,"q_load":0b010001,
  "r_change_br":0b010010,"a_change_br":0b010011,"mayday":0b010100,"wypnt_rchd":0b011000,"wypnt_unrchd":0b011001,"cllsn":0b011010 }
'''lat and long is longitude and latitiude ,emg is emergency,,q is query , fin is final , r is request , a is accept , d is deny,sc= service culing ,br=bearing,wypnt is waypoint'''
#deny change of airspeed absent